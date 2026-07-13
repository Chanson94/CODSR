import os
import gc
import lpips
import argparse
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from codsr import CODSR_gen_Stage2_Pipeline, CODSR_reg
from dataloaders.realsr_dataset import PairedSROnlineTxtDataset  

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from models.sam2 import load_seg_model
from attn_utils.tc_loss_utils import get_grounding_loss_by_list

from nltk import pos_tag

from dataloaders.Realesrgan_offline_all_HQ_dataset_stage2 import LocalImageDataset_selectedv2

def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--output_dir", default='experience/codsr_stage2')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=10000,)
    parser.add_argument("--warm_up_steps", type=int, default=3000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_codsr_stage2", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument('--dataset_txt_paths_list_val', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list_val', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--neg_prompt", default="", type=str)

    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")

    

    # lora setting
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--lora_rank_unet", default=4, type=int)
    # parser.add_argument("--lora_rank", default=32, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    # val freq
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--need_val', type=bool, default=True)

    parser.add_argument("--vae_path", default=None, type=str)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--seg_model", default="gsam", type=str)

    # parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-checkpoint", default="seg_model/gsam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--null_text_ratio", default=0.3, type=float)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def collate_fn(data):
    lq_images = torch.stack([example["lq_image"] for example in data])
    images = torch.stack([example["image"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])
    
    gt_path = [example["gt_path"] for example in data]

    return {
        "lq_images": lq_images,
        "images": images,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "gt_path": gt_path,
    }

def extract_nouns(original_list):

    words = original_list[0].split(', ')
    tagged_words = pos_tag(words)

    filtered_words = [word for word, pos in tagged_words 
                    if pos not in ['JJ', 'JJR', 'JJS']]
    # print(filtered_words)
    result = [', '.join(filtered_words)]

    return result
    
def generate_attention_map(per_layer_heatmaps, layer_name, x_src):

    heat = per_layer_heatmaps[layer_name][0:1] 

    B, C, H, W = x_src.shape

    heat_up = F.interpolate(heat.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[0, 0]  # [H, W]

    return heat_up

def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = CODSR_gen_Stage2_Pipeline(args)
     # set vae adapter
    model_gen.model.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.model.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others', 'default_encoder_alignment', 'default_decoder_alignment', 'default_others_alignment'])
    model_gen.model.set_train()
 
    seg_model = load_seg_model(args, accelerator.device)


    # init VSDLoss model
    model_reg = CODSR_reg(args=args, accelerator=accelerator)
    model_reg.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            # model_gen.model.unet.enable_xformers_memory_efficient_attention()
            model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            model_reg.unet_update.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.model.unet.enable_gradient_checkpointing()
        model_reg.unet_fix.enable_gradient_checkpointing()
        model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # make the optimizer
    layers_to_opt = []

    for n, _p in model_gen.model.unet.named_parameters():
        # if "alignment" in n:
        if "alignment" in n and "attn2" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    layers_to_opt_reg = []
    for n, _p in model_reg.unet_update.named_parameters():
        if "lora" in n:
            layers_to_opt_reg.append(_p)
    optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)
    
    img_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/HR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/HR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/HR_crops']
    lq_img_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/LR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/LR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/LR_crops']

    train_dataset = LocalImageDataset_selectedv2(img_file=[img_file_path, lq_img_file_path], image_size=512, t_drop_rate=0.2)
    
    
    print("Dataset loading finished")

    dl_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=6,
        pin_memory=True,
    )

    # dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="val", args=args)
    # dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    
    # init vlm model
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_vlm = ram(pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    model_vlm.eval()
    model_vlm.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    model_gen, model_reg, seg_model,optimizer, optimizer_reg, dl_train, dl_val, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, seg_model,optimizer, optimizer_reg, dl_train, dl_val, lr_scheduler, lr_scheduler_reg
    )

    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        args.dataset_txt_paths_list_val = str(args.dataset_txt_paths_list_val)
        args.dataset_prob_paths_list_val = str(args.dataset_prob_paths_list_val)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    
    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):

                x_src = batch["lq_images"]
                x_tgt = batch["images"]
                B, C, H, W = x_src.shape
                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                batch["prompt"] = [f'{each_caption}' for each_caption in caption]

                ### Extract Nouns
                nouns_prompts = extract_nouns(batch["prompt"])
                word_list = nouns_prompts[0].split(',')
                clean_words_list = [word.strip() for word in word_list if word.strip()]

                attn_maps = []

                # forward pass
                x_tgt_pred, latents_denoised, prompt_embeds, neg_prompt_embeds = model_gen(x_src.detach(), prompt=batch["prompt"], num_inference_steps=1, args=args)
                
                if len(clean_words_list) != 0:
                    for word in clean_words_list:
                        per_layer_heatmaps = model_gen.model.build_cross_attention_maps_for_word(
                            prompts=batch["prompt"],
                            word=word,
                            base_hw=(64, 64)
                        )

                        layer_name = next(iter(per_layer_heatmaps.keys())) 

                        attention_map = generate_attention_map(
                            per_layer_heatmaps, 
                            layer_name=layer_name, 
                            x_src=x_src, 
                        ).unsqueeze(0)

                        attn_maps.append(attention_map)

                        # attn_map_pil = transforms.ToPILImage()(attention_map)
                        # os.makedirs(os.path.join(args.output_dir, f'eval/attn_map_{global_step}_{str(accelerator.device)}'), exist_ok=True)
                        # attn_map_pil.save(os.path.join(args.output_dir, f'eval/attn_map_{global_step}_{str(accelerator.device)}/'+word+'.png'))


                if len(clean_words_list) != 0:

                    # # token_loss, pixel_loss
                    mask_list = seg_model.get_binary_mask(x_tgt[0] * 0.5 + 0.5, clean_words_list)
                    mask_list_on_device = [torch.from_numpy(mask).unsqueeze(0).to(accelerator.device) for mask in mask_list]

                    if len(mask_list) == len(attn_maps):
                        attn_loss_dict = get_grounding_loss_by_list(
                            _gt_seg_list=mask_list_on_device,
                            input_attn_map_ls=attn_maps,
                        )
                        loss_area = attn_loss_dict["token_loss"]
                        loss_pixel = attn_loss_dict["pixel_loss"]

                    else:
                        raise NotImplementedError
                        
                # lambda and total loss
                lambda_l2 = 1
                lambda_lpips = 2
                lambda_vsd = 2
                lambda_area = 0.5   
                lambda_pixel = 0.025

                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * lambda_lpips

                # KL loss
                if torch.cuda.device_count() > 1:
                    loss_kl = model_reg.module.distribution_matching_loss(latents=latents_denoised, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * lambda_vsd 
                else:
                    loss_kl = model_reg.distribution_matching_loss(latents=latents_denoised, prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, args=args) * lambda_vsd
                
                loss = loss_l2 + loss_lpips + loss_kl
               
                if len(clean_words_list) != 0:
                    
                    loss_area = loss_area * lambda_area
                    loss_pixel = loss_pixel * lambda_pixel

                    loss = loss + loss_area + loss_pixel

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                diff loss: let lora model closed to generator 
                """
                if torch.cuda.device_count() > 1:
                    loss_d = model_reg.module.diff_loss(latents=latents_denoised, prompt_embeds=prompt_embeds, args=args)
                else:
                    loss_d = model_reg.diff_loss(latents=latents_denoised, prompt_embeds=prompt_embeds, args=args)
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_reg.parameters(), args.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

                ### release 
                model_gen.model.cross_attn_store.clear()
                del attn_maps
                if len(clean_words_list) != 0:
                    del mask_list, mask_list_on_device
                # del tc, heat_map, attn_map, attn_maps, mask_list, mask_list_on_device
                if global_step % 100 == 1:
                    gc.collect()
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    
                    if len(clean_words_list) != 0:
                        if not isinstance(loss_area, float):
                            logs["loss_area"] = loss_area.detach().item()
                        if not isinstance(loss_pixel, float):
                            logs["loss_pixel"] = loss_pixel.detach().item()
          
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model (also save at the final step before exit)
                    if global_step % args.checkpointing_steps == 1 or global_step >= args.max_train_steps:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen.model).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if args.need_val:
                        if global_step % args.val_freq == 1:
                            # l_l2, l_lpips = [], []
                            val_count = 0
                            for step, batch in enumerate(dl_val):
                                if val_count <= 0:
                                    x_src = batch["conditioning_pixel_values"]
                                    x_tgt = batch["output_pixel_values"]
                                    B, C, H, W = x_src.shape
                                    assert B == 1, "Use batch size 1 for eval."
                                    with torch.no_grad():
                                        
                                        x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                                        caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                                        batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                                        # forward pass
                                        x_tgt_pred = model_gen(x_src, prompt=batch["prompt"], num_inference_steps=1, args=args)[0]

                                        x_src = x_src.cpu().detach() * 0.5 + 0.5
                                        x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
                                        x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5

                                        combined = torch.cat([x_src, x_tgt_pred, x_tgt], dim=3)
                                        output_pil = transforms.ToPILImage()(combined[0])
                                        outf = os.path.join(args.output_dir, f"eval/val_{global_step}.png")
                                        output_pil.save(outf)
                                        val_count += 1

                            gc.collect()
                            torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

if __name__ == "__main__":
    args = parse_args()
    main(args)
