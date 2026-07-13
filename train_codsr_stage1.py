import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from diffusers.optimization import get_scheduler

from codsr import CODSR_gen_Stage1
from dataloaders.realsr_dataset import PairedSROnlineTxtDataset  

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from dataloaders.Realesrgan_offline_all_HQ_dataset import LocalImageDataset

from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument("--output_dir", default='experience/codsr_stage1')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=14000,)
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
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_codsr_stage1", help="The name of the wandb project to log to.")
    # parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    # parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
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
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    # val freq
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--need_val', type=bool, default=True)


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

    model_gen = CODSR_gen_Stage1(args).cuda()
    model_gen.set_train()

    
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # set vae adapter
    model_gen.vae.set_adapter(['default_encoder'])

    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()


    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # discriminator of GAN
    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    # make the optimizer
    layers_to_opt = []

    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    layers_to_opt += list(model_gen.sft.parameters())

    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)

    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    ### new datasets
    img_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/HR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/HR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/HR_crops']
    lq_img_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/LR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/LR_crops', '/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/LR_crops']
    face_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/FFHQ/HR_crops']
    face_lq_file_path = ['/home/notebook/data/group/ch/Datasets/High_quality_training_data/FFHQ/LR_crops_1']
    
    train_dataset = LocalImageDataset(img_file=[img_file_path, lq_img_file_path], face_file = [face_file_path, face_lq_file_path], image_size=512, t_drop_rate=0.2)
    print("Dataset loading finished")

    dl_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=6,
        pin_memory=True,
    )

    print("before:", len(dl_train))

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
    model_gen, net_disc, optimizer, optimizer_disc, dl_train, dl_val, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        model_gen, net_disc, optimizer, optimizer_disc, dl_train, dl_val, lr_scheduler, lr_scheduler_disc
    )

    print("after:", len(dl_train))
    
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
        # args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        # args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        args.dataset_txt_paths_list_val = str(args.dataset_txt_paths_list_val)
        args.dataset_prob_paths_list_val = str(args.dataset_prob_paths_list_val)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, net_disc]
            with accelerator.accumulate(*m_acc):

                for p in net_disc.parameters():
                    p.requires_grad = False
                    
                x_src = batch["lq_images"]
                x_tgt = batch["images"]
                B, C, H, W = x_src.shape
                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                batch["prompt"] = [f'{each_caption}' for each_caption in caption]

                # forward pass
                x_tgt_pred = model_gen(x_src.detach(), batch=batch, args=args)

                # Reconstruction losss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean")
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean()

                lambda_l2 = 1
                lambda_lpips = 2
            
                loss_l2 = loss_l2 * lambda_l2
                loss_lpips = loss_lpips * lambda_lpips
                loss =  loss_l2 + loss_lpips

                if global_step < args.warm_up_steps:

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                else:
                    """
                    Generator loss: fool the discriminator
                    """
                    lossG = net_disc(x_tgt_pred, for_G=True).mean()
                    lambda_gan = 0.2
                    lossG = lossG * lambda_gan

                    loss =  loss + lossG 

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    """
                    Discriminator loss: fake image vs real image
                    """
                    for p in net_disc.parameters():
                        p.requires_grad = True

                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * lambda_gan
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * lambda_gan

                    lossD = lossD_real + lossD_fake
                    accelerator.backward(lossD)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    # log all the losses
                    if global_step > args.warm_up_steps:
                        logs["lossG"] = lossG.detach().item()
                        logs["lossD"] = lossD.detach().item()

                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model (also save at the final step before exit)
                    if global_step % args.checkpointing_steps == 1 or global_step >= args.max_train_steps:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

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
                                        x_tgt_pred = model_gen(x_src, batch=batch, args=args)

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
