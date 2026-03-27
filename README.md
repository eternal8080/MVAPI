## MVAPI: Multimodal Visual Attention Prompting for Image-aware Reasoning

MVAPI is a training-free visual prompting framework that leverages attention mechanisms from vision-language models (CLIP / SigLIP) to generate question-aware image masks. By highlighting the most relevant visual regions for a given question, MVAPI helps multimodal large language models (MLLMs) focus on critical information, improving performance on visual mathematical reasoning tasks.

### Key Features

- **Training-free**: No fine-tuning required — uses pre-trained CLIP/SigLIP models directly
- **Attention-guided masking**: Combines CLS-token attention maps and MLP token maps to produce fine-grained visual masks
- **Multi-image support**: Handles both single-image and multi-image reasoning scenarios with cross-image attention weighting
- **Flexible prompting**: Supports multiple prompt strategies (masked, hint, iterative, step-by-step) via configurable templates
- **Multi-benchmark evaluation**: Ready-to-use scripts for GeoQA, MathVista, MathVision, VCBench, MMSI, ReMI, and MV-MATH

### Project Structure

```
MVAPI/
├── DatasetManager/          # Unified dataset loading module
│   └── DatasetManager/
│       ├── dataloader.py    # Dataset classes for all supported benchmarks
│       └── __init__.py
├── MVAPI_CLIP/              # CLIP-based mask generation
│   ├── clip_prs/            # Modified open_clip with hook support
│   │   └── utils/           # Model factory, tokenizer, transforms, etc.
│   ├── hook.py              # Projected Residual Stream (PRS) hook logger
│   ├── main_geoqa.py        # Mask generation for GeoQA
│   ├── main_mathvista.py    # Mask generation for MathVista
│   ├── main_MV-MATH.py      # Mask generation for MV-MATH (multi-image)
│   ├── main_mmsi.py         # Mask generation for MMSI
│   ├── main_ReMI.py         # Mask generation for ReMI
│   ├── main_vcbench.py      # Mask generation for VCBench
│   ├── heatmap1.py          # Single-image heatmap visualization
│   └── heatmap2.py          # Heatmap visualization with custom colormaps
├── MVAPI_SigLIP/            # SigLIP-based mask generation
│   ├── hook_siglip.py       # SigLIP attention hook logger
│   ├── main_geoqa.py        # Mask generation for GeoQA
│   ├── main_mathvista.py    # Mask generation for MathVista
│   ├── main-MV-math.py      # Mask generation for MV-MATH
│   ├── main_mmsi.py         # Mask generation for MMSI
│   ├── main_ReMI.py         # Mask generation for ReMI
│   └── main_vcbench.py      # Mask generation for VCBench
└── Prompts/
    └── prompts.json         # Prompt templates for different strategies
```

### Installation

#### Requirements

- Python >= 3.8
- PyTorch >= 1.13
- CUDA-compatible GPU

#### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/MVAPI.git
cd MVAPI

# Install dependencies
pip install torch torchvision
pip install timm einops ftfy scipy pillow pandas numpy transformers datasets
pip install opencv-python scikit-image scikit-learn tqdm matplotlib

# Install the DatasetManager module
cd DatasetManager
pip install -e .
cd ..
```

### Usage

#### 1. Prepare Datasets

Download the datasets and update the default paths in `DatasetManager/DatasetManager/dataloader.py`, or pass paths as arguments at runtime.

Supported datasets:
| Dataset | Type | Source |
|---------|------|--------|
| GeoQA | Geometry QA | [GeoQA](https://github.com/chen-judge/GeoQA) |
| MathVista | Math Visual QA | [MathVista](https://mathvista.github.io/) |
| VCBench | Visual Counting | [VCBench](https://alibaba-damo-academy.github.io/VCBench/) |
| MMSI | Multi-image SI | [MMSI](https://runsenxu.com/projects/MMSI_Bench) |
| ReMI | Reasoning with MI | [ReMI](https://huggingface.co/datasets/mehrankazemi/ReMI) |
| MV-MATH | Multi-view Math | [MV-MATH](https://drive.google.com/file/d/1odVEBTs7-xhXf2hmNyHzoBRa3YnMqOa_/view?usp=drive_link) |

#### 2. Generate Masks with CLIP

```bash
cd MVAPI_CLIP

# Single-image dataset (e.g., GeoQA)
python main_geoqa.py \
    --dataset GeoQA \
    --model_name ViT-L-14-336 \
    --layer_index 22 \
    --enhance_coe 5 \
    --kernel_size 3 \
    --grayscale 100 \
    --output_folder your/output/path

# Multi-image dataset (e.g., MV-MATH)
python main_MV-MATH.py \
    --dataset custom \
    --model_name ViT-L-14-336 \
    --layer_index 22 \
    --output_folder your/output/path
```

#### 3. Generate Masks with SigLIP

```bash
cd MVAPI_SigLIP

python main_geoqa.py \
    --dataset GeoQA \
    --layer_index 22 \
    --enhance_coe 5 \
    --kernel_size 3 \
    --grayscale 100 \
    --output_folder your/output/path
```


### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | CLIP model variant | `ViT-L-14-336` |
| `--layer_index` | Transformer layer to extract attention from | `22` |
| `--enhance_coe` | Enhancement coefficient for attention sigmoid | `5` |
| `--kernel_size` | Smoothing convolution kernel size | `3` |
| `--grayscale` | Background grayscale value for masked regions | `100` |
| `--interpolate_method_name` | Mask upsampling method | `LANCZOS` |

### How It Works

1. **Attention Extraction**: Hooks into a specified transformer layer of CLIP/SigLIP to capture the attention weights (CLS→patch) and MLP outputs
2. **Text-guided Projection**: Projects attention maps and MLP token maps onto the text embedding direction via dot product, producing question-relevant spatial maps
3. **Mask Fusion**: Normalizes and enhances both maps, then fuses them using the formula: `mask = attn + mlp - attn * mlp`
4. **Image Blending**: Upsamples the mask to the original image resolution and blends it with a gray background, highlighting relevant regions

### Prompt Strategies

The `Prompts/prompts.json` file defines several prompting strategies:

- **`empty`**: Original image with the raw question
- **`masked`**: Masked image with the raw question
- **`masked_hint`**: Masked image with a hint about visible regions
- **`hint`**: Original image with an answer hint
- **`iterative`**: Original image with self-refinement prompt
- **`sbs`**: Step-by-step reasoning prompt

### License

The CLIP PRS component is licensed under [CC BY-NC 4.0](MVAPI_CLIP/clip_prs/LICENSE.txt).
