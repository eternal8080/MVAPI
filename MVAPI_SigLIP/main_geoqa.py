# main_siglip2.py
import os, sys, argparse
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from hook_siglip import SigLIPHookLogger
from torch.nn.functional import normalize, sigmoid


project_root = 'your/path/MVAPI'
sys.path.append(project_root)
from DatasetManager.DatasetManager.dataloader import get_data, Disposable_Dataloader


def toImg(t):
    return T.ToPILImage()(t)

def invtrans(mask, image, method=Image.BICUBIC):
    return mask.resize(image.size, method)

def merge(mask, image, grap_scale=200):
    gray = np.ones((image.size[1], image.size[0], 3)) * grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None]  + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image

def normalize_tensor(mat, method="max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError

def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    return sigmoid(mat * coe).clamp(0, 1)

def apply_minimum_cutoff(mask, threshold=0.5):
    return torch.clamp(mask, min=threshold)

def merge_mask(cls_mask, patch_mask, kernel_size=3, enhance_coe=10):
    cls_mask = normalize_tensor(cls_mask, "min")
    cls_mask = enhance(cls_mask, coe=enhance_coe)

    patch_mask = normalize_tensor(patch_mask, "max")
    padding = (kernel_size - 1) // 2
    conv = torch.nn.Conv2d(1, 1, kernel_size, padding=padding, padding_mode="replicate", bias=False)
    conv.weight.data.fill_(1.0 / (kernel_size ** 2))
    conv = conv.to(cls_mask.device)

    cls_mask = conv(cls_mask.unsqueeze(0))[0]
    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = normalize_tensor(cls_mask + patch_mask - cls_mask * patch_mask, "min")
    return apply_minimum_cutoff(mask, threshold=0.5)

def blend_mask(image, cls_mask, patch_mask, key, enhance_coe, kernel_size, interpolate_method, grayscale, folder):
    mask = merge_mask(cls_mask, patch_mask, kernel_size, enhance_coe)
    mask = toImg(mask.detach().cpu().unsqueeze(0))
    mask = invtrans(mask, image, method=interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB"), grayscale).convert("RGB")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(key)
    merged_image.save(os.path.join(folder, filename))

    return merged_image

def get_model_and_hook(model_name="google/siglip2-large-patch16-384", cache_dir=None, layer_index=23):
    model = AutoModel.from_pretrained(model_name, device_map="auto", attn_implementation="sdpa", cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    hook_logger = SigLIPHookLogger(model, device="cuda", layer_index=layer_index)
    return model, processor, hook_logger

def gen_mask(model, processor, hook_logger, device, image_pils, questions):
    texts = [f"This is a photo of {q}." for q in questions]
    inputs = processor(images=image_pils, text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)

    hook_logger.reinit()
    with torch.no_grad():
        outputs = model(**inputs)
        text_embeds = normalize(outputs.text_embeds, dim=-1)
        attn_map, mlp_map = hook_logger.finalize(normalize_target=text_embeds)

    return image_pils, attn_map, mlp_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GeoQA")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layer_index", type=int, default=22)
    parser.add_argument("--range", type=int, nargs=2, default=(0, 1109))
    parser.add_argument("--output_folder", type=str, default="your/path/MVAPI/MVAPI_SigLIP/result_mvmath")
    parser.add_argument("--interpolate_method_name", type=str, default="LANCZOS")
    parser.add_argument("--enhance_coe", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--grayscale", type=int, default=100)
    args = parser.parse_args()

    exp_folder = f"{args.output_folder}/SIGLIP_{args.dataset}_{args.layer_index}"
    device = "cuda"

    model, processor, hook_logger = get_model_and_hook(layer_index=args.layer_index)
    dataset = get_data(args.dataset)
    dataset_loader = Disposable_Dataloader(dataset, args.batch_size, args.range)

    interpolate_method = getattr(Image, args.interpolate_method_name)
    mask_image_folder = os.path.join(exp_folder, f"{args.enhance_coe}_{args.kernel_size}_{args.interpolate_method_name}_{args.grayscale}_concise")
    mask_image_folder = os.path.join(mask_image_folder, "input_image")
    os.makedirs(mask_image_folder, exist_ok=True)

    with tqdm(total=len(dataset), desc="Processing Batches") as pbar_outer:
        for keys, image_path_or_pil_images, questions, image_paths in dataset_loader:
            questions = questions * len(image_paths[0])
            keys = keys * len(image_paths[0])

            image_pils = [Image.open(p).convert("RGB") for p in image_paths[0]]
            image_pils, cls_masks, patch_masks = gen_mask(model, processor, hook_logger, device, image_pils, questions)

            for key, image, cls_mask, patch_mask, image_path in zip(keys, image_pils, cls_masks, patch_masks, image_paths[0]):
                with tqdm(total=1, desc=f"Processing {key}", leave=False) as pbar_inner:
                    blend_mask(image, cls_mask, patch_mask, image_path, args.enhance_coe, args.kernel_size, interpolate_method, args.grayscale, mask_image_folder)
                    pbar_inner.update(1)
            pbar_outer.update(1)
