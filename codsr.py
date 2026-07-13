import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler

from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from peft import LoraConfig

from my_utils.vaehook import VAEHook

from diffusers.utils.peft_utils import set_weights_and_activate_adapters

import time 

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ChannelwiseSobel(torch.nn.Module):
    def __init__(self, mode='accurate'):
        super().__init__()
        
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32)

        kernel_y = torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=torch.float32)
        
        # Register as buffer (important!)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

        self.mode = mode
        
    def forward(self, x):
        kernel_x = self.kernel_x.repeat(x.size(1), 1, 1, 1).to(x.device)
        kernel_y = self.kernel_y.repeat(x.size(1), 1, 1, 1).to(x.device)

        padded_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        padded_y = F.pad(x, (1, 1, 1, 1), mode='replicate')
        
        grad_x = F.conv2d(padded_x, kernel_x, padding=0, groups=x.size(1))
        grad_y = F.conv2d(padded_y, kernel_y, padding=0, groups=x.size(1))
        
        if self.mode == 'accurate':
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        else:
            magnitude = torch.abs(grad_x) + torch.abs(grad_y)

        magnitude = torch.mean(magnitude, dim=1, keepdim=True).repeat(1, x.size(1), 1, 1)

        return magnitude

class SFTLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.SFT_scale_conv1 = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.SFT_shift_conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.SFT_shift_conv1 = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        self.conv_act = nn.SiLU()
    
    ### Unet 320 latent
    def forward(self, cond):
        scale = self.SFT_scale_conv1(self.conv_act(self.SFT_scale_conv0(cond)))
        shift = self.SFT_shift_conv1(self.conv_act(self.SFT_shift_conv0(cond)))
        return scale, shift

def initialize_vae(args):
    print("Initialize Vae LoRA !!!!")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=args.lora_rank_vae, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder

def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    # l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj", "time"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank_unet, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank_unet, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank_unet, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

class CODSR_test(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        self.sobel_layer = ChannelwiseSobel(mode='fast')

        ### SFT Modulation
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        self.sft = SFTLayer(192,320)

        model_ckp = torch.load(args.codsr_path, map_location="cuda")
        self.load_ckpt(model_ckp)

        # merge lora
        set_weights_and_activate_adapters(self.unet, ["default_encoder_alignment", "default_decoder_alignment", "default_others_alignment"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.text_encoder.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([100], device="cuda").long()  
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

        self.sobel_layer.to("cuda", dtype=self.weight_dtype)
        self.sft.to("cuda", dtype=self.weight_dtype)

    def load_ckpt(self, model):

        _sft = self.sft.state_dict()
        for k in model["state_dict_sft"]:
            _sft[k] = model["state_dict_sft"][k]
        self.sft.load_state_dict(_sft)

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

        # load unet lora
        self.lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules_default"])
        self.lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules_default"])
        self.lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules_default"])

        self.lora_conf_encoder_sem = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules_sam"])
        self.lora_conf_decoder_sem = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules_sam"])
        self.lora_conf_others_sem = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules_sam"])
        
        self.unet.add_adapter(self.lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(self.lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(self.lora_conf_others, adapter_name="default_others")

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_unet"][n])

        # Merge and save unet weights
        set_weights_and_activate_adapters(self.unet, ["default_encoder", "default_decoder", "default_others"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        # Add semantic adapters
        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_alignment")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_alignment")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_alignment")

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_unet"][n])

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def rgb_to_gray(self, rgb_map):

        r, g, b = rgb_map[:, 0], rgb_map[:, 1], rgb_map[:, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        return gray.unsqueeze(1)
    
    def GraygradToWeight_Patchwise_Sobel(self, grad_map, target_hw8=None):
        b, c, h, w = grad_map.shape
        device = grad_map.device
        dtype  = grad_map.dtype

        block_size = 16

        pad_h = (-h) % block_size
        pad_w = (-w) % block_size
        if pad_h or pad_w:
            grad_map = F.pad(grad_map, (0, pad_w, 0, pad_h), mode='replicate')

        patch_avg = F.avg_pool2d(grad_map, kernel_size=block_size, stride=block_size)

        result_patch = torch.zeros_like(patch_avg)
        low_mask = patch_avg <= 0.15
        mid_mask = (patch_avg > 0.15) & (patch_avg <= 0.25)
        high_mask = patch_avg > 0.25

        result_patch[low_mask] = 0.3
        result_patch[mid_mask] = 7.0 * (patch_avg[mid_mask] - 0.15) + 0.3
        result_patch[high_mask] = 1.0


        expanded_weight = result_patch.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # [B,1,H/8',W/8']

        if target_hw8 is not None:
            H8, W8 = target_hw8
        else:
            H8, W8 = h // 8, w // 8

        expanded_weight = expanded_weight[:, :, :H8, :W8]

        return expanded_weight.to(device=device, dtype=dtype)

    def eps_to_mu_coeff(self, scheduler, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        coeff = beta_prod_t ** (0.5) / alpha_prod_t ** (0.5)
        return coeff
    
    # @perfcount
    @torch.no_grad()
    def forward(self, lq, prompt):

        encoded_control = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor

        ## based on lq
        cond = self.unshuffle(lq.to(self.weight_dtype))
        unet_params = self.sft(cond)

        # # cal space weight based on grayscale
        gray_map = self.rgb_to_gray(lq.to(self.weight_dtype)*0.5+0.5)
        gradient_result = self.sobel_layer(gray_map)

        H8, W8 = encoded_control.shape[-2], encoded_control.shape[-1]
        mix_weight_Sobel = self.GraygradToWeight_Patchwise_Sobel(gradient_result, target_hw8=(H8, W8))

        ## add noise
        noise_a = torch.randn_like(encoded_control) * mix_weight_Sobel
        lq_latent = self.noise_scheduler.add_noise(encoded_control, noise_a, self.timesteps)

        ### Coeff
        coeff = self.eps_to_mu_coeff(self.noise_scheduler, lq_latent, self.timesteps)
        unet_params = tuple(x * 1/coeff for x in unet_params)

        prompt_embeds = self.encode_prompt([prompt])

        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds, modulation_params=unet_params).sample
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")

            unet_dtype  = next(self.unet.parameters()).dtype
            unet_device = next(self.unet.parameters()).device

            lq_latent     = lq_latent.to(unet_device, dtype=unet_dtype)
            prompt_embeds = prompt_embeds.to(unet_device, dtype=unet_dtype)

            H_lat, W_lat = lq_latent.shape[-2:]
            def _resize_param_to_latent(p):
                if torch.is_tensor(p):
                    if p.dim() == 4: 
                        if p.shape[-2:] != (H_lat, W_lat):
                            p = torch.nn.functional.interpolate(p, size=(H_lat, W_lat),
                                                                mode="bilinear", align_corners=False)
                    elif p.dim() == 2: 
                        p = p[:, :, None, None]
                    return p.to(unet_device, dtype=unet_dtype)
                return p

            if isinstance(unet_params, (list, tuple)):
                unet_params_lat = tuple(_resize_param_to_latent(p) for p in unet_params)
            else:
                unet_params_lat = _resize_param_to_latent(unet_params)

            tile_size    = min(tile_size, min(H_lat, W_lat))
            stride       = tile_size - tile_overlap
            grid_rows    = max(1, (H_lat - tile_overlap + stride - 1) // stride) 
            grid_cols    = max(1, (W_lat  - tile_overlap + stride - 1) // stride) 

            def _tile_offset(i, n, total, size, overlap):
                if i < n - 1:
                    return max(i * size - overlap * i, 0)
                else:
                    return total - size

            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(unet_device, dtype=unet_dtype)

            noise_pred   = torch.zeros_like(lq_latent, dtype=unet_dtype, device=unet_device)
            contributors = torch.zeros_like(lq_latent, dtype=unet_dtype, device=unet_device)

            for r in range(grid_rows):
                for c in range(grid_cols):
                    y0 = _tile_offset(r, grid_rows, H_lat, tile_size, tile_overlap)
                    x0 = _tile_offset(c, grid_cols, W_lat,  tile_size, tile_overlap)
                    y1, x1 = y0 + tile_size, x0 + tile_size

                    tile_latent = lq_latent[:, :, y0:y1, x0:x1]

                    def _slice_param(p):
                        if torch.is_tensor(p):
                            if p.dim() == 4:
                                return p[:, :, y0:y1, x0:x1]
                            elif p.dim() == 2:
                                return p[:, :, None, None]
                        return p

                    if isinstance(unet_params_lat, (list, tuple)):
                        tile_params = tuple(_slice_param(p) for p in unet_params_lat)
                    else:
                        tile_params = _slice_param(unet_params_lat)

                    timesteps = self.timesteps.to(unet_device)
                    with torch.cuda.amp.autocast(enabled=(unet_dtype == torch.float16), dtype=torch.float16):
                        tile_out = self.unet(
                            tile_latent, timesteps,
                            encoder_hidden_states=prompt_embeds,
                            modulation_params=tile_params
                        ).sample

                    noise_pred[:, :, y0:y1, x0:x1]   += tile_out * tile_weights
                    contributors[:, :, y0:y1, x0:x1] += tile_weights

                    del tile_latent, tile_params, tile_out
                    torch.cuda.empty_cache()

            model_pred = noise_pred / contributors.clamp_min(1e-6)

        
        x_denoised = encoded_control + coeff*(noise_a - model_pred)
        
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

class CODSR_gen_Stage1(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args) 
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.lora_rank_vae = self.args.lora_rank_vae 
        self.lora_rank_unet = self.args.lora_rank_unet

        ### SFT Modulation
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        self.sft = SFTLayer(192,320)
        self.sft.to("cuda")

        self.sobel_layer = ChannelwiseSobel(mode='fast')
        self.sobel_layer.to("cuda")

        self._rng = None

        self.unet.to("cuda")
        self.vae.to("cuda")

        self.text_encoder.requires_grad_(False)

    def set_train(self):

        self.vae.train()
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
         
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        ### sft layer
        self.sft.train()
        self.sft.requires_grad_(True)

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def rgb_to_gray(self, rgb_map):

        r, g, b = rgb_map[:, 0], rgb_map[:, 1], rgb_map[:, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        return gray.unsqueeze(1)
    
    def _get_rng(self, device):
        if self._rng is None:
            import torch.distributed as dist
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(100_000 * self.rank + int(10 * time.time()))
        
        return self._rng
    
    def GraygradToWeight_Patchwise_Sobel(self, grad_map):
        b, c, h, w = grad_map.shape
        block_size = 16
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size

        patches = grad_map.reshape(b, c, num_blocks_h, block_size, num_blocks_w, block_size)
        
        ### [b, c , 32, 32]
        patch_avg = patches.mean(dim=[3, 5])

        result_patch = torch.zeros_like(patch_avg)
        low_mask = patch_avg <= 0.15
        mid_mask = torch.logical_and(patch_avg > 0.15, patch_avg <= 0.25)
        high_mask = patch_avg > 0.25

        result_patch[low_mask] = 0.3
        result_patch[mid_mask] = 7.0 * ( patch_avg[mid_mask] - 0.15 ) + 0.3
        result_patch[high_mask] = 1.0

        expanded_weight = result_patch.repeat_interleave(2, dim=2)
        expanded_weight = expanded_weight.repeat_interleave(2, dim=3)

        return expanded_weight.repeat(1, 1, 1, 1)
    
    def eps_to_mu_coeff(self, scheduler, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        coeff = beta_prod_t ** (0.5) / alpha_prod_t ** (0.5)
        return coeff
    
    def forward(self, c_t, batch=None, args=None):
        
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        ## based on lq
        cond = self.unshuffle(c_t)
        unet_params = self.sft(cond)

        ### sample timestep
        bsz = encoded_control.shape[0]
        rng = self._get_rng(encoded_control.device)
        self.timesteps = torch.randint(
            50, 251, (bsz,),
            device=encoded_control.device,
            dtype=torch.long,
            generator=rng
        )

        # cal space weight based on grayscale
        gray_map = self.rgb_to_gray(c_t*0.5+0.5)
        gradient_result = self.sobel_layer(gray_map)
        mix_weight_Sobel = self.GraygradToWeight_Patchwise_Sobel(gradient_result)
        
        noise_a = torch.randn_like(encoded_control) * mix_weight_Sobel
        noise_encoded_control = self.noise_scheduler.add_noise(encoded_control, noise_a, self.timesteps)

        ### Coeff
        coeff = self.eps_to_mu_coeff(self.noise_scheduler, noise_encoded_control, self.timesteps)
        unet_params = tuple(x * 1/coeff for x in unet_params)

        # calculate prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])

        model_pred = self.unet(noise_encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32), modulation_params=unet_params).sample

        x_denoised = encoded_control + coeff*(noise_a - model_pred)

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        
        sd["state_dict_sft"] = {k: v for k, v in self.sft.state_dict().items()}

        torch.save(sd, outf)

class CODSR_gen_Stage2_Pipeline():
    def __init__(self, args):
        super().__init__()
        self.model = CODSR_gen_Stage2(args)
        self.vae = self.model.vae
        self.unet = self.model.unet
        self.tokenizer = self.model.tokenizer
    
    def __call__(self, lq, prompt, num_inference_steps, args):
        
        self.check_inputs(
            lq,
            prompt,
        )
        hq = self.model(lq, prompt, args)

        return hq 
    
    def check_inputs(
        self,
        lq,
        prompt,
    ):
        if prompt is None or lq is None:
            raise ValueError(
                f"input images or prompt is nan!"
            )
        
from attn_recorder import CrossAttnCaptureProcessor
from collections import OrderedDict

class CODSR_gen_Stage2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        if args.resume_path:
            self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
            self.vae.requires_grad_(False)
            self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")
            self.unet.requires_grad_(False)

            ### SFT Modulation
            self.unshuffle = nn.PixelUnshuffle(downscale_factor=8)
            self.sft = SFTLayer(192,320)
            self.sft.to("cuda")

            self.sobel_layer = ChannelwiseSobel(mode='fast')
            self.sobel_layer.to("cuda")

            model_ckp = torch.load(args.resume_path, map_location="cuda")
            self.load_ckpt(model_ckp)

        else:
            assert 0
        
        # ------- recorder

        self.cross_attn_store = OrderedDict()

        def _inject_cross_attn_recorder(unet, store):
            procs = {}
            for name, proc in unet.attn_processors.items():
                if name.endswith("attn2.processor"):
                    procs[name] = CrossAttnCaptureProcessor(store, name) 
                else:
                    procs[name] = proc
            unet.set_attn_processor(procs)

        _inject_cross_attn_recorder(self.unet, self.cross_attn_store)

        # --------------

        self.unet.to("cuda")
        self.vae.to("cuda")

        self.text_encoder.requires_grad_(False)

        self.neg_prompt = ["painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"]
        
        self._rng = None    
    
    def load_ckpt(self, model):

        ### load lora 
        self.lora_vae_modules_encoder = model["vae_lora_encoder_modules"]
        self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = model["unet_lora_encoder_modules"], model["unet_lora_decoder_modules"], model["unet_lora_others_modules"]

        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_unet"][n])
        # self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        # self.vae.set_adapter(['default_encoder'])

        # others
        self.lora_rank_vae = model["rank_vae"]
        self.lora_rank_unet = model["rank_unet"]

        _sft = self.sft.state_dict()
        for k in model["state_dict_sft"]:
            _sft[k] = model["state_dict_sft"][k]
        self.sft.load_state_dict(_sft)

        # set sam LoRA
        self.lora_unet_modules_encoder_sam, self.lora_unet_modules_decoder_sam, self.lora_unet_others_sam = model["unet_lora_encoder_modules"], model["unet_lora_decoder_modules"], model["unet_lora_others_modules"]

        lora_conf_encoder_sam = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder_sam = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others_sam = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])

        self.unet.add_adapter(lora_conf_encoder_sam, adapter_name="default_encoder_alignment")
        self.unet.add_adapter(lora_conf_decoder_sam, adapter_name="default_decoder_alignment")
        self.unet.add_adapter(lora_conf_others_sam, adapter_name="default_others_alignment")

    def set_train(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            # if "alignment" in n:
            if "alignment" in n and "attn2" in n:
                _p.requires_grad = True
        self.sft.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def rgb_to_gray(self, rgb_map):

        r, g, b = rgb_map[:, 0], rgb_map[:, 1], rgb_map[:, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        return gray.unsqueeze(1)
    
    def GraygradToWeight_Patchwise_Sobel(self, grad_map):
        b, c, h, w = grad_map.shape
        block_size = 16
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size

        patches = grad_map.reshape(b, c, num_blocks_h, block_size, num_blocks_w, block_size)
        
        ### [b, c , 32, 32]
        patch_avg = patches.mean(dim=[3, 5])

        result_patch = torch.zeros_like(patch_avg)
        low_mask = patch_avg <= 0.15
        mid_mask = torch.logical_and(patch_avg > 0.15, patch_avg <= 0.25)
        high_mask = patch_avg > 0.25


        result_patch[low_mask] = 0.3
        result_patch[mid_mask] = 7.0 * ( patch_avg[mid_mask] - 0.15 ) + 0.3
        result_patch[high_mask] = 1.0

        expanded_weight = result_patch.repeat_interleave(2, dim=2)
        expanded_weight = expanded_weight.repeat_interleave(2, dim=3)

        return expanded_weight.repeat(1, 1, 1, 1)
    
    def eps_to_mu_coeff(self, scheduler, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        coeff = beta_prod_t ** (0.5) / alpha_prod_t ** (0.5)
        return coeff
    
    def _get_rng(self, device):
        if self._rng is None:
            import torch.distributed as dist
            self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._rng = torch.Generator(device=device)
        self._rng.manual_seed(100_000 * self.rank + int(10 * time.time()))
        
        return self._rng
    
    def _latent_hw_from_length(self, L, base_hw):

        H8, W8 = base_hw
        candidates = [(H8, W8), (H8//2, W8//2), (H8//4, W8//4)]
        for h, w in candidates:
            if h*w == L:
                return (h, w)
        # Fallback: nearest factor decomposition
        import math
        w = int(round((L * (W8 / H8)) ** 0.5))
        h = L // w
        return (h, w)

    def build_token_heatmaps_from_store(
        self,
        store,                       
        tokenizer,                   
        prompts: list,               
        target_token_or_word: str,  
        base_hw,                     
        reduce_heads: bool = True   
    ):

        device = next(self.unet.parameters()).device
        per_layer = {}

        with torch.no_grad():
            enc = tokenizer(
                prompts,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = enc.input_ids  # [B, 77]

            target_enc = tokenizer(
                target_token_or_word,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target_ids = target_enc.input_ids[0].tolist()  # e.g., [320, 1234]

            token_positions_per_sample = []
            for b in range(input_ids.shape[0]):
                ids = input_ids[b].tolist()
                pos = []
                L = len(ids)
                T = len(target_ids)
                for i in range(L - T + 1):
                    if ids[i:i+T] == target_ids:
                        pos.extend(list(range(i, i+T)))
                if len(pos) == 0:
                    pos = list(range(L))
                token_positions_per_sample.append(pos)


        for layer_name, attn_probs in store.items():

            B, Hh, Q, K = attn_probs.shape
            h, w = self._latent_hw_from_length(Q, base_hw)

            heat = []

            for b in range(B):
                pos = token_positions_per_sample[b]
                hb = attn_probs[b, :, :, pos].mean(dim=-1)  # [heads, Q]
                heat.append(hb.unsqueeze(0))
            heat = torch.cat(heat, dim=0)  # [B, heads, Q]

            if reduce_heads:
                heat = heat.mean(dim=1)     # [B, Q]
                heat = heat.view(B, h, w)   # [B, h, w]
            else:
                heat = heat.view(B, Hh, h, w)

            per_layer[layer_name] = heat

        return per_layer

    def build_cross_attention_maps_for_word(self, prompts, word, base_hw):
        return self.build_token_heatmaps_from_store(
            store=self.cross_attn_store,
            tokenizer=self.tokenizer,
            prompts=prompts,
            target_token_or_word=word,
            base_hw=base_hw,
            reduce_heads=True
        )
    
    def forward(self, c_t, prompt=None, args=None):
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        ## based on lq
        cond = self.unshuffle(c_t)
        unet_params = self.sft(cond)

        ### Add noise
        bsz = encoded_control.shape[0]
        rng = self._get_rng(encoded_control.device)
        self.timesteps = torch.randint(
            50, 251, (bsz,),
            device=encoded_control.device,
            dtype=torch.long,
            generator=rng
        )

        # # cal space weight based on grayscale
        gray_map = self.rgb_to_gray(c_t*0.5+0.5)
        gradient_result = self.sobel_layer(gray_map)
        mix_weight_Sobel = self.GraygradToWeight_Patchwise_Sobel(gradient_result)

        noise_a = torch.randn_like(encoded_control) * mix_weight_Sobel
        noise_encoded_control = self.noise_scheduler.add_noise(encoded_control, noise_a, self.timesteps)

        ### Coeff
        coeff = self.eps_to_mu_coeff(self.noise_scheduler, noise_encoded_control, self.timesteps)
        unet_params = tuple(x * 1/coeff for x in unet_params)

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(prompt)
        neg_prompt_embeds = self.encode_prompt(self.neg_prompt)

        model_pred = self.unet(noise_encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32), modulation_params=unet_params).sample
        
        x_denoised = encoded_control + coeff*(noise_a - model_pred)
        
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules_default"], sd["unet_lora_decoder_modules_default"], sd["unet_lora_others_modules_default"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["unet_lora_encoder_modules_sam"], sd["unet_lora_decoder_modules_sam"], sd["unet_lora_others_modules_sam"] =\
            self.lora_unet_modules_encoder_sam, self.lora_unet_modules_decoder_sam, self.lora_unet_others_sam
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "conv_out" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        
        sd["state_dict_sft"] = {k: v for k, v in self.sft.state_dict().items()}
        
        torch.save(sd, outf)

class CODSR_reg(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(args)

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            ## concat
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss =  F.mse_loss(latents, (latents - grad).detach())

        return loss


