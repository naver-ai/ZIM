# ZIM: Zero-Shot Image Matting for Anything

[Beomyoung Kim](https://beomyoung-kim.github.io/), Chanyong Shin, [Joonhyun Jeong](https://bestdeveloper691.github.io/), Hyungsik Jung, Se-Yun Lee, Sewhan Chun, [Dong-Hyun Hwang](https://hwangdonghyun.github.io/), Joonsang Yu<br>

<sub>NAVER Cloud, ImageVision</sub><br />

[![Paper](https://img.shields.io/badge/Paper-arxiv-red)](https://arxiv.org/pdf/2411.00626)
[![Page](https://img.shields.io/badge/Project_page-blue)](https://naver-ai.github.io/ZIM) 	
[![🤗 demo](https://img.shields.io/badge/Hugging%20Face-Demo-FFD21E?logo=huggingface&logo)](https://huggingface.co/spaces/naver-iv/ZIM_Zero-Shot-Image-Matting)
[![🤗 Dataset](https://img.shields.io/badge/Hugging%20Face%20Dataset-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/datasets/naver-iv/MicroMat-3K)
[![🤗 Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/models?search=naver-iv/zim)
[![🤗 Collection](https://img.shields.io/badge/Hugging%20Face%20Collection-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/collections/naver-iv/zim-6739a32211b7efeeed83b674)

![Teaser](https://github.com/naver-ai/ZIM/releases/download/asset-v1/amg.gif)
![Teaser](https://github.com/naver-ai/ZIM/releases/download/asset-v1/teaser.png)

## Introduction

The recent segmentation foundation model, Segment Anything Model (SAM), exhibits strong zero-shot segmentation capabilities, but it falls short in generating fine-grained precise masks. To address this limitation, we propose a novel zero-shot image matting model, called ZIM, with two key contributions: First, we develop a label converter that transforms segmentation labels into detailed matte labels, constructing the new SA1B-Matte dataset without costly manual annotations. Training SAM with this dataset enables it to generate precise matte masks while maintaining its zero-shot capability. Second, we design the zero-shot matting model equipped with a hierarchical pixel decoder to enhance mask representation, along with a prompt-aware masked attention mechanism to improve performance by enabling the model to focus on regions specified by visual prompts. We evaluate ZIM using the newly introduced MicroMat-3K test set, which contains high-quality micro-level matte labels. Experimental results show that ZIM outperforms existing methods in fine-grained mask generation and zero-shot generalization. Furthermore, we demonstrate the versatility of ZIM in various downstream tasks requiring precise masks, such as image inpainting and 3D NeRF. Our contributions provide a robust foundation for advancing zero-shot matting and its downstream applications across a wide range of computer vision tasks. 

![Model overview](https://github.com/naver-ai/ZIM/releases/download/asset-v1/method_overview.png)

## Updates    
- 2025.07.24: ZIM has been accepted to **ICCV 2025** as a **Highlight Paper**!
- 2024.11.04: official ZIM code update


# Installation

Install the required packages with the command below:
```bash
pip install zim_anything
```

or
```bash
git clone https://github.com/naver-ai/ZIM.git
cd ZIM; pip install -e .
```

To enable GPU acceleration, please install the compatible `onnxruntime-gpu` package based on your environment settings (CUDA and CuDNN versions), following the instructions in the [onnxruntime installation docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).


## Demo

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/naver-iv/ZIM_Zero-Shot-Image-Matting)  We provide a Gradio demo code in `demo/gradio_demo.py`. You can run our model demo locally by running:

```bash
python demo/gradio_demo.py
```

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/naver-iv/ZIM_demo_with_SAM)  In addition, we provide a Gradio demo code `demo/gradio_demo_comparison.py` to qualitatively compare ZIM with SAM:

```bash
python demo/gradio_demo.py
```
## Getting Started

After the installation step is done, you can utilize our model in just a few lines as below. `ZimPredictor` is compatible with `SamPredictor`, such as `set_image()` or `predict()`.
```python
from zim_anything import zim_model_registry, ZimPredictor

backbone = "vit_l"
ckpt_p = "results/zim_vit_l_2092"

model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

predictor = ZimPredictor(model)
predictor.set_image(<image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

We also provide code for generating masks for an entire image and visualization:

```python
from zim_anything import zim_model_registry, ZimAutomaticMaskGenerator
from zim_anything.utils import show_mat_anns

backbone = "vit_l"
ckpt_p = "results/zim_vit_l_2092"

model = zim_model_registry[backbone](checkpoint=ckpt_p)
if torch.cuda.is_available():
    model.cuda()

mask_generator = ZimAutomaticMaskGenerator(model)
masks = mask_generator.generate(<image>)  # Automatically generated masks
masks_vis = show_mat_anns(<image>, masks)  # Visualize masks
```

Additionally, masks can be generated for images from the command line:
```bash
bash script/run_amg.sh
```

We provide Pretrained-weights of ZIM.
|   MODEL ZOO  | Link |
| :------:  | :------:  |
| zim_vit_b | [download](https://huggingface.co/naver-iv/zim-anything-vitb/tree/main/zim_vit_b_2043) |
| zim_vit_l | [download](https://huggingface.co/naver-iv/zim-anything-vitl/tree/main/zim_vit_l_2092) |


## Dataset Preparation

### 1) MicroMat-3K Dataset
![MicroMat-3K](https://github.com/naver-ai/ZIM/releases/download/asset-v1/qualitative_micromat.png)
We introduce a new test set named MicroMat-3K, to evaluate zero-shot interactive matting models. It consists of 3,000 high-resolution images paired with micro-level matte labels, providing a comprehensive benchmark for testing various matting models under different levels of detail.

Downloading **MicroMat-3K** dataset is available [here](https://github.com/naver-ai/ZIM/releases/download/testset-v1/MicroMat3K.tar) or [huggingface](https://huggingface.co/datasets/naver-iv/MicroMat-3K)

#### 1-1) Dataset structure

Dataset structure should be as follows:
```bash
└── /path/to/dataset/MicroMat3K
    ├── img
    │   ├── 0001.png
    ├── matte
    │   ├── coarse
    │   │   ├── 0001.png
    │   └── fine
    │       ├── 0001.png
    ├── prompt
    │   ├── coarse
    │   │   ├── 0001.png
    │   └── fine
    │       ├── 0001.png
    └── seg
        ├── coarse
        │   ├── 0001_01.json
        └── fine
            ├── 0001_01.json
```

#### 1-2) Prompt file configuration

Prompt file configuration should be as follows:
```json
{
    "point": [[x1, y1, 1], [x2, y2, 0], ...],   # 1: Positive, 0: Negative prompt
    "bbox": [x1, y1, x2, y2]                    # [X, Y, X, Y] format
}
```

## Evaluation

We provide an evaluation script, which includes a comparison with SAM, in `script/run_eval.sh`. Make sure the dataset structure is prepared.

First, modify `data_root` in `script/run_eval.sh`
```bash
...
data_root="/path/to/dataset/"
...
```

Then, run evaluation script file.
```bash
bash script/run_eval.sh
```

The evaluation result on the MicroMat-3K dataset would be as follows:

![Table](https://github.com/naver-ai/ZIM/releases/download/asset-v1/Table1.png)


## How To Cite

```
@article{kim2024zim,
  title={ZIM: Zero-Shot Image Matting for Anything},
  author={Kim, Beomyoung and Shin, Chanyong and Jeong, Joonhyun and Jung, Hyungsik and Lee, Se-Yun and Chun, Sewhan and Hwang, Dong-Hyun and Yu, Joonsang},
  journal={arXiv preprint arXiv:2411.00626},
  year={2024}
}
```

## License

```
ZIM
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)  
```
