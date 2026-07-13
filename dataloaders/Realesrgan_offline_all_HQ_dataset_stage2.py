import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
# from transformers.models.llava_next.image_processing_llava_next import LlavaNextImageProcessor
import torch.nn.functional as F
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from PIL import Image
import json
from transformers import CLIPImageProcessor
from torch import nn
from torchvision import transforms
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .realesrgan import RealESRGAN_degradation
import cv2
import random
from glob import glob
from collections import OrderedDict
import yaml
from PIL import Image
import time
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image
def exists(x):
    return x is not None


def list_paired_image_paths(hr_roots, lr_roots):
    nature_paths = []
    nature_lr_paths = []
    for img_path_idx in hr_roots:
        img_path_list = sorted(glob(os.path.join(img_path_idx, "**", "*.png"), recursive=True))
        nature_paths += img_path_list
    for lq_img_path_idx in lr_roots:
        lq_img_path_list = sorted(glob(os.path.join(lq_img_path_idx, "**", "*.png"), recursive=True))
        nature_lr_paths += lq_img_path_list
    return nature_paths, nature_lr_paths


def get_offline_mask_path(gt_path, mask_root):
    parts = gt_path.replace("\\", "/").split("/")
    if "HR_crops" not in parts:
        raise ValueError(f"Cannot resolve offline mask path from: {gt_path}")
    idx = parts.index("HR_crops")
    rel = "/".join(parts[idx - 1 :])
    rel = rel.rsplit(".", 1)[0] + ".npz"
    return os.path.join(mask_root, rel)


def paired_center_crop_np(pil_img, pil_lr_img, crop_size=512):
    crop_pad_size = crop_size // 4
    h, w = pil_lr_img.shape[0:2]

    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        pil_lr_img = cv2.copyMakeBorder(pil_lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        pad_h = max(0, crop_size - h * 4)
        pad_w = max(0, crop_size - w * 4)
        pil_img = cv2.copyMakeBorder(pil_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    h, w = pil_lr_img.shape[0:2]
    if pil_lr_img.shape[0] > crop_pad_size or pil_lr_img.shape[1] > crop_pad_size:
        top = (h - crop_pad_size) // 2
        left = (w - crop_pad_size) // 2
        pil_lr_img = pil_lr_img[top : top + crop_pad_size, left : left + crop_pad_size, ...]
        pil_img = pil_img[top * 4 : (top + crop_pad_size) * 4, left * 4 : (left + crop_pad_size) * 4, ...]
    else:
        top = 0
        left = 0

    return pil_img, pil_lr_img, top, left


class LocalImageDataset_selectedv2(data.Dataset):
    def __init__(self, 
                img_file = None,
                yml_kernel = None,
                image_size=512,
                tokenizer=None,
                tokenizer_2=None,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
                t_drop_rate=0.05
        ):
        super(LocalImageDataset_selectedv2, self).__init__()

        self.resize_bak = resize_bak

        self.crop_size = image_size

        self.t_drop_rate = t_drop_rate

        nature_paths = []
        nature_lr_paths = []

        self.data_types = ['nature']

        for img_path_idx in img_file[0]:
            img_path_list = sorted(glob(os.path.join(img_path_idx, '**', '*.png'), recursive=True))
            nature_paths += img_path_list

        for lq_img_path_idx in img_file[1]:
            lq_img_path_list = sorted(glob(os.path.join(lq_img_path_idx, '**', '*.png'), recursive=True))
            nature_lr_paths += lq_img_path_list

        self.data_collection = {'nature': (np.array(nature_paths), np.array(nature_lr_paths))}
        self.data_lens = {'nature': len(nature_paths)}
        print(self.data_lens)

        self.data_lr_lens = {'nature': len(nature_lr_paths)}
        print(self.data_lens)

        self.datatypes_lens = [len(nature_paths)]
        self.cumulative_lens = np.cumsum([0] + self.datatypes_lens)

    def __getitem__(self, index):

        data_type_idx = np.where(self.cumulative_lens <= index )[0][-1]

        data_type = self.data_types[data_type_idx]
        index = index - self.cumulative_lens[data_type_idx]

        crop_pad_size = self.crop_size
        # load image
        img_path = self.data_collection[data_type][0][index]
        lq_img_path = self.data_collection[data_type][1][index]

        gt_path = img_path

        image = Image.open(img_path).convert('RGB')
        
        if 'FFHQ' in lq_img_path:
            if random.random() < 0.5:
                lq_img_path = lq_img_path.replace('LR_crops_1', 'LR_crops_2')

        lq_image = Image.open(lq_img_path).convert('RGB')

        if 'FFHQ' in img_path:
            random_size = random.randint(128, 192)
            lq_image = lq_image.resize((random_size, random_size), Image.BICUBIC)  
            image = image.resize((int(random_size * 4), int(random_size * 4)), Image.BICUBIC)  

        w, h = lq_image.size
        pil_img = np.array(image)  
        pil_lr_img = np.array(lq_image)   
        pil_img, pil_lr_img = augment([pil_img, pil_lr_img], hflip=True, rotation=False)

        crop_pad_size = self.crop_size // 4 #1024 #
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            pil_lr_img = cv2.copyMakeBorder(pil_lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            pad_h = max(0, self.crop_size - h * 4)
            pad_w = max(0, self.crop_size - w * 4)
            pil_img = cv2.copyMakeBorder(pil_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)


        # crop
        if pil_lr_img.shape[0] > crop_pad_size or pil_lr_img.shape[1] > crop_pad_size:
            h, w = pil_lr_img.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            
            pil_lr_img = pil_lr_img[top : top + crop_pad_size, left : left + crop_pad_size, ...]
            pil_img = pil_img[top * 4 : (top + crop_pad_size) * 4, left * 4: (left + crop_pad_size) * 4, ...]

        else:
            top = 0
            left = 0
        
        lq_image = Image.fromarray(pil_lr_img)
        mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        lr_w, lr_h = lq_image.size
        lq_image = lq_image.resize((lr_w * 4, lr_h * 4), mode)  
        
        image = Image.fromarray(pil_img)
        original_size = torch.tensor([h * 4, w * 4])
        crop_coords_top_left = torch.tensor([top * 4, left * 4])

        GT_image_t = np.asarray(image)/255.
        LR_image_t = np.asarray(lq_image)/255.

        GT_image_t, LR_image_t = img2tensor([GT_image_t, LR_image_t], bgr2rgb=False, float32=True)
        LR_image_t = LR_image_t * 2.0 - 1.0
        GT_image_t =  GT_image_t * 2.0 - 1.0

        rand_num = random.random()
        if rand_num < self.t_drop_rate:
            text = ""
    
        return {
            'lq_image': LR_image_t,
            "image": GT_image_t,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([crop_pad_size, crop_pad_size]),
            'gt_path': gt_path,
        }

    def __len__(self):
        total_length = 0
        for key, value in self.data_lens.items():
            total_length += value
        return total_length


class LocalImageDataset_selectedv2_offline(data.Dataset):
    """Offline stage2 dataset: load precomputed caption/words/masks with center crop."""

    def __init__(
        self,
        img_file=None,
        mask_root=None,
        image_size=512,
        random_flip=True,
        t_drop_rate=0.05,
        require_mask=True,
    ):
        super(LocalImageDataset_selectedv2_offline, self).__init__()
        assert mask_root is not None, "mask_root is required for offline dataset"
        self.mask_root = mask_root
        self.crop_size = image_size
        self.random_flip = random_flip
        self.t_drop_rate = t_drop_rate
        self.require_mask = require_mask

        nature_paths, nature_lr_paths = list_paired_image_paths(img_file[0], img_file[1])
        self.data_types = ["nature"]
        self.data_collection = {"nature": (np.array(nature_paths), np.array(nature_lr_paths))}
        self.data_lens = {"nature": len(nature_paths)}
        print(self.data_lens)
        self.datatypes_lens = [len(nature_paths)]
        self.cumulative_lens = np.cumsum([0] + self.datatypes_lens)

    def __getitem__(self, index):
        data_type_idx = np.where(self.cumulative_lens <= index)[0][-1]
        data_type = self.data_types[data_type_idx]
        index = index - self.cumulative_lens[data_type_idx]

        img_path = self.data_collection[data_type][0][index]
        lq_img_path = self.data_collection[data_type][1][index]
        gt_path = img_path

        mask_path = get_offline_mask_path(gt_path, self.mask_root)
        if not os.path.isfile(mask_path):
            if self.require_mask:
                raise FileNotFoundError(f"Missing offline mask: {mask_path}")
            caption = ""
            words = []
            masks_np = np.zeros((0, self.crop_size, self.crop_size), dtype=np.float32)
        else:
            payload = np.load(mask_path, allow_pickle=True)
            caption = str(payload["caption"])
            words = [str(w) for w in payload["words"].tolist()] if payload["words"].size > 0 else []
            masks_np = payload["masks"]
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]
            if masks_np.shape[0] == 0:
                masks_np = np.zeros((0, self.crop_size, self.crop_size), dtype=np.float32)

        image = Image.open(img_path).convert("RGB")
        lq_image = Image.open(lq_img_path).convert("RGB")
        pil_img = np.array(image)
        pil_lr_img = np.array(lq_image)

        pil_img, pil_lr_img, top, left = paired_center_crop_np(
            pil_img, pil_lr_img, crop_size=self.crop_size
        )

        do_flip = self.random_flip and random.random() < 0.5
        if do_flip:
            pil_img = np.fliplr(pil_img).copy()
            pil_lr_img = np.fliplr(pil_lr_img).copy()
            if masks_np.shape[0] > 0:
                masks_np = np.flip(masks_np, axis=-1).copy()

        lq_image = Image.fromarray(pil_lr_img)
        mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        lr_w, lr_h = lq_image.size
        lq_image = lq_image.resize((lr_w * 4, lr_h * 4), mode)
        image = Image.fromarray(pil_img)

        h, w = pil_lr_img.shape[0] * 4, pil_lr_img.shape[1] * 4
        original_size = torch.tensor([h, w])
        crop_coords_top_left = torch.tensor([top * 4, left * 4])
        crop_pad_size = self.crop_size // 4

        GT_image_t = np.asarray(image) / 255.0
        LR_image_t = np.asarray(lq_image) / 255.0
        GT_image_t, LR_image_t = img2tensor([GT_image_t, LR_image_t], bgr2rgb=False, float32=True)
        LR_image_t = LR_image_t * 2.0 - 1.0
        GT_image_t = GT_image_t * 2.0 - 1.0

        if random.random() < self.t_drop_rate:
            caption = ""
            words = []
            masks_np = np.zeros((0, self.crop_size, self.crop_size), dtype=np.float32)

        masks = [torch.from_numpy(m.astype(np.float32)) for m in masks_np]

        return {
            "lq_image": LR_image_t,
            "image": GT_image_t,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([crop_pad_size, crop_pad_size]),
            "gt_path": gt_path,
            "prompt": caption,
            "words": words,
            "masks": masks,
        }

    def __len__(self):
        total_length = 0
        for key, value in self.data_lens.items():
            total_length += value
        return total_length
