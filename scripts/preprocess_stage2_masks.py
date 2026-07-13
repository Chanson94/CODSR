import argparse
import os
import sys

import numpy as np
import torch
from nltk import pos_tag
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloaders.Realesrgan_offline_all_HQ_dataset_stage2 import (
    get_offline_mask_path,
    list_paired_image_paths,
    paired_center_crop_np,
)
from models.sam2 import load_seg_model


def parse_args():
    parser = argparse.ArgumentParser(description="Offline SAM2 mask preprocessing for CODSR stage2.")
    parser.add_argument(
        "--hr_roots",
        type=str,
        nargs="+",
        default=[
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/HR_crops",
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/HR_crops",
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/HR_crops",
        ],
    )
    parser.add_argument(
        "--lr_roots",
        type=str,
        nargs="+",
        default=[
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV2K/LR_crops",
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/DIV8K/LR_crops",
            "/home/notebook/data/group/ch/Datasets/High_quality_training_data/LSDIR/LR_crops",
        ],
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="/home/notebook/data/group/ch/Datasets/High_quality_training_data/stage2_offline_masks",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--ram_path", type=str, default="preset/models/ram_swin_large_14m.pth")
    parser.add_argument("--seg_model", default="gsam", type=str)
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-checkpoint", default="seg_model/gsam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1, help="Exclusive end index. -1 means all.")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    return parser.parse_args()


def extract_nouns_from_caption(caption):
    words = caption.split(", ")
    tagged_words = pos_tag(words)
    filtered_words = [word for word, pos in tagged_words if pos not in ["JJ", "JJR", "JJS"]]
    clean_words = [w.strip() for w in filtered_words if w.strip()]
    return clean_words


def main():
    args = parse_args()
    os.makedirs(args.mask_root, exist_ok=True)

    hr_paths, lr_paths = list_paired_image_paths(args.hr_roots, args.lr_roots)
    assert len(hr_paths) == len(lr_paths), f"HR/LR count mismatch: {len(hr_paths)} vs {len(lr_paths)}"

    end_idx = len(hr_paths) if args.end_idx < 0 else min(args.end_idx, len(hr_paths))
    start_idx = max(0, args.start_idx)
    hr_paths = hr_paths[start_idx:end_idx]
    lr_paths = lr_paths[start_idx:end_idx]
    print(f"Processing {len(hr_paths)} pairs [{start_idx}:{end_idx}] -> {args.mask_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seg_model = load_seg_model(args, device)

    from ram.models.ram_lora import ram
    from ram import inference_ram as inference

    ram_transforms = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model_vlm = ram(
        pretrained=args.ram_path,
        pretrained_condition=None,
        image_size=384,
        vit="swin_l",
    )
    model_vlm.eval()
    model_vlm.to(device, dtype=torch.float16)

    for hr_path, lr_path in tqdm(list(zip(hr_paths, lr_paths)), desc="offline_masks"):
        out_path = get_offline_mask_path(hr_path, args.mask_root)
        if args.skip_existing and os.path.isfile(out_path):
            continue

        image = Image.open(hr_path).convert("RGB")
        lq_image = Image.open(lr_path).convert("RGB")
        pil_img = np.array(image)
        pil_lr_img = np.array(lq_image)
        pil_img, pil_lr_img, _, _ = paired_center_crop_np(pil_img, pil_lr_img, crop_size=args.image_size)

        hq_pil = Image.fromarray(pil_img)
        hq_tensor = transforms.ToTensor()(hq_pil).to(device)  # [0,1], CxHxW

        with torch.no_grad():
            x_ram = ram_transforms(hq_tensor.unsqueeze(0)).to(dtype=torch.float16)
            caption = inference(x_ram, model_vlm)[0]

        clean_words = extract_nouns_from_caption(caption)
        if len(clean_words) == 0:
            masks = np.zeros((0, args.image_size, args.image_size), dtype=np.float32)
            words = np.array([], dtype=object)
        else:
            mask_list = seg_model.get_binary_mask(
                hq_tensor,
                clean_words,
                image_size=(args.image_size, args.image_size),
            )
            masks = np.stack([m.astype(np.float32) for m in mask_list], axis=0)
            words = np.array(clean_words, dtype=object)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            caption=np.array(caption),
            words=words,
            masks=masks,
            hr_path=np.array(hr_path),
        )

    print("Done.")


if __name__ == "__main__":
    main()
