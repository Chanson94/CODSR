import torch.nn as nn
import torch
from seg_model.gsam2.sam2.build_sam import build_sam2
from seg_model.gsam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os

def load_seg_model(args, device):
    model = SegModel(args, device)
    model.eval()
    return model.to(device)

class SegModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        # build SAM2 image predictor
        self.sam2_model = build_sam2(
            args.sam2_model_config,
            args.sam2_checkpoint,
            device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # build grounding dino from huggingface
        # self.processor = AutoProcessor.from_pretrained(args.grounding_model)
        # self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device)
        path = "preset/models/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(device)
        for param in self.parameters():
            param.requires_grad = False

    def get_binary_mask(
            self,
            image,
            word_list,
            max_words=100,
            max_boxes_per_word=2,
            image_size=(512, 512),
        ):

        import gc

        clean = []
        seen = set()
        for w in word_list:
            if not w:
                continue
            t = w.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            clean.append(t)
            seen.add(key)
            if len(clean) >= max_words:
                break

        image_pil = transforms.ToPILImage()(image.detach().cpu())
        self.sam2_predictor.set_image(np.array(image_pil.convert("RGB")))

        zero = np.zeros(image_size, dtype=np.float32)
        word2mask = {}
        use_cuda = (self.device.type == "cuda")


        for word in clean:
            try:
                inputs = self.processor(
                    images=image_pil,
                    text=word.lower().rstrip(".") + ".",
                    return_tensors="pt",
                ).to(self.grounding_model.device)

                with torch.inference_mode():
                    outputs = self.grounding_model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image_pil.size[::-1]],  # (H, W)
                )

                boxes = results[0].get("boxes", None)
                scores = results[0].get("scores", None)

                if (boxes is None) or (boxes.numel() == 0):
                    word2mask[word] = zero
                    del inputs, outputs, results
                    continue

                if isinstance(scores, torch.Tensor) and scores.numel() == boxes.shape[0]:
                    topk = min(max_boxes_per_word, boxes.shape[0])
                    sel_idx = torch.topk(scores, k=topk, largest=True).indices
                    boxes = boxes.index_select(0, sel_idx)
                else:
                    boxes = boxes[:max_boxes_per_word]

                input_boxes = boxes.detach().float().cpu().numpy()

                if use_cuda:
                    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
                else:

                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()

                with torch.inference_mode():
                    with autocast_ctx:
                        masks, _, _ = self.sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )

                if masks is None:
                    word2mask[word] = zero
                else:
                    m = masks
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    if m.ndim == 4:  # [N,1,H,W]
                        m = m.squeeze(1)
                    merged = np.any(m > 0.5, axis=0).astype(np.float32)
                    if merged.shape != image_size:
                        merged = cv2.resize(
                            merged, image_size[::-1],
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.float32)
                    word2mask[word] = merged

                del inputs, outputs, results, boxes, scores, masks

            except torch.cuda.OutOfMemoryError:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    inputs = self.processor(
                        images=image_pil,
                        text=word.lower().rstrip(".") + ".",
                        return_tensors="pt",
                    ).to(self.grounding_model.device)

                    with torch.inference_mode():
                        outputs = self.grounding_model(**inputs)

                    results = self.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=0.4,
                        text_threshold=0.3,
                        target_sizes=[image_pil.size[::-1]],
                    )

                    boxes = results[0].get("boxes", None)
                    if (boxes is None) or (boxes.numel() == 0):
                        word2mask[word] = zero
                    else:
                        boxes = boxes[:1]
                        input_boxes = boxes.detach().float().cpu().numpy()

                        if use_cuda:
                            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
                        else:
                            from contextlib import nullcontext
                            autocast_ctx = nullcontext()

                        with torch.inference_mode():
                            with autocast_ctx:
                                masks, _, _ = self.sam2_predictor.predict(
                                    point_coords=None, point_labels=None,
                                    box=input_boxes, multimask_output=False
                                )

                        if masks is None:
                            word2mask[word] = zero
                        else:
                            m = masks
                            if isinstance(m, torch.Tensor):
                                m = m.detach().cpu().numpy()
                            if m.ndim == 4:
                                m = m.squeeze(1)
                            merged = np.any(m > 0.5, axis=0).astype(np.float32)
                            if merged.shape != image_size:
                                merged = cv2.resize(
                                    merged, image_size[::-1],
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(np.float32)
                            word2mask[word] = merged

                    del inputs, outputs, results, boxes, masks

                except Exception:
                    word2mask[word] = zero
            except Exception:
                word2mask[word] = zero

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        masks_list = []
        for w in word_list:
            t = (w or "").strip()
            masks_list.append(word2mask.get(t, zero))
        return masks_list
