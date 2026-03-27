import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from clip_prs.utils.factory import create_model_and_transforms, get_tokenizer
from hook import hook_prs_logger
from torch.nn import functional as F


def normalize(mat, method="max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError


def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0, 1)
    return mat


def apply_minimum_cutoff(mask: torch.Tensor, threshold: float = 0.5):
    return torch.clamp(mask, min=threshold)


def save_heatmap(tensor: torch.Tensor, save_path: str, cmap='YlGn'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    array = tensor.detach().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(array, cmap=cmap, interpolation='nearest')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_model(model_name="ViT-L-14-336", layer_index=23):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = create_model_and_transforms(model_name, pretrained='openai')
    model.to(device)
    model.eval()
    tokenizer = get_tokenizer(model_name)
    prs = hook_prs_logger(model, device, layer_index)
    return model, prs, preprocess, device, tokenizer


def gen_mask(model, prs, preprocess, device, tokenizer, image_path, question):
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image_pil)[None].to(device)
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image_tensor, attn_method='head', normalize=False)
        attentions, mlps = prs.finalize(representation)

    text_tensor = tokenizer([question]).to(device)
    class_embedding = F.normalize(model.encode_text(text_tensor), dim=-1)

    attention_map = attentions[:, 0, 1:, :]
    attention_map = torch.einsum('bnd,bd->bn', attention_map, class_embedding)
    HW = int(attention_map.shape[1] ** 0.5)
    attention_map = attention_map.view(1, HW, HW)

    token_map = torch.einsum('bnd,bd->bn', mlps[:, 0, :, :], class_embedding)
    token_map = token_map.view(1, HW, HW)

    return image_pil, attention_map[0], token_map[0]


def merge_mask(cls_mask, patch_mask, kernel_size=3, enhance_coe=10):
    cls_mask = normalize(cls_mask, "min")
    cls_mask = enhance(cls_mask, coe=enhance_coe)
    patch_mask = normalize(patch_mask, "max")

    assert kernel_size % 2 == 1
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2,
                           padding_mode="replicate", stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size ** 2
    conv = conv.to(cls_mask.device)

    cls_mask = conv(cls_mask.unsqueeze(0).unsqueeze(0))[0, 0]
    patch_mask = conv(patch_mask.unsqueeze(0).unsqueeze(0))[0, 0]

    mask = normalize(cls_mask + patch_mask - cls_mask * patch_mask, "min")
    mask = torch.clamp(mask, 0, 1)
    mask = apply_minimum_cutoff(mask, threshold=0.0)
    return mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="your/path/to/input_image.jpg", help="Path to input image")
    parser.add_argument("--question", type=str, default="rectangle ABCD, AB=4 and BC=3. Point $E$ starts from the midpoint of $B C$ and moves counterclockwise along the sides of the rectangle until it reaches the midpoint of $A D$. Let the distance traveled by point $E$ be $x$ and the area of triangle $A B E$ be $y$. Which of the following graphs correctly represents the functional relationship between $y$ and $x$?", help="Question for the image")
    parser.add_argument("--output_path", type=str, default="your/path/to/output_heatmap.png", help="Path to save heatmap")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--layer_index", type=int, default=22)
    parser.add_argument("--enhance_coe", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for apply_minimum_cutoff")  # 添加threshold参数
    args = parser.parse_args()

    model, prs, preprocess, device, tokenizer = get_model(args.model_name, args.layer_index)
    image_pil, cls_mask, patch_mask = gen_mask(model, prs, preprocess, device, tokenizer, args.image_path, args.question)
    fused_mask = merge_mask(cls_mask, patch_mask, kernel_size=args.kernel_size, enhance_coe=args.enhance_coe)
    fused_mask = apply_minimum_cutoff(fused_mask, threshold=args.threshold)  # 使用args.threshold
    save_heatmap(fused_mask, args.output_path, cmap='YlGn')
