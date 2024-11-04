# ZIM

**ZIM: Zero-Shot Image Matting for Anything** <br />
[Beomyoung Kim](https://beomyoung-kim.github.io/), Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, Joonsang Yu<br>

<sub>NAVER Cloud, ImageVision</sub><br />

[![Paper](https://img.shields.io/badge/Paper-arxiv)]([https://arxiv.org](https://arxiv.org/pdf/2411.00626)) 	
[![Page](https://img.shields.io/badge/Project_page-blue)](https://naver-ai.github.io/ZIM) 	
[![Demo](https://img.shields.io/badge/Demo-yellow)](https://huggingface.co/spaces/naver-iv/ZIM_Zero-Shot-Image-Matting)
[![Data](https://img.shields.io/badge/Data-gray)](https://huggingface.co/datasets/naver-iv/MicroMat-3K)

![Teaser]([./assets/teaser.png](https://github.com/naver-ai/ZIM/releases/download/asset-v1/teaser.png))
![Model overview]([./assets/method_overview.png](https://github.com/naver-ai/ZIM/releases/download/asset-v1/method_overview.png))

## Introduction

The recent segmentation foundation model, Segment Anything Model (SAM), exhibits strong zero-shot segmentation capabilities, but it falls short in generating fine-grained precise masks. To address this limitation, we propose a novel zero-shot image matting model, called ZIM, with two key contributions: First, we develop a label converter that transforms segmentation labels into detailed matte labels, constructing the new SA1B-Matte dataset without costly manual annotations. Training SAM with this dataset enables it to generate precise matte masks while maintaining its zero-shot capability. Second, we design the zero-shot matting model equipped with a hierarchical pixel decoder to enhance mask representation, along with a prompt-aware masked attention mechanism to improve performance by enabling the model to focus on regions specified by visual prompts. We evaluate ZIM using the newly introduced MicroMat-3K test set, which contains high-quality micro-level matte labels. Experimental results show that ZIM outperforms existing methods in fine-grained mask generation and zero-shot generalization. Furthermore, we demonstrate the versatility of ZIM in various downstream tasks requiring precise masks, such as image inpainting and 3D NeRF. Our contributions provide a robust foundation for advancing zero-shot matting and its downstream applications across a wide range of computer vision tasks. 

## Updates    
- 2024.11.04: official ZIM code update


## Getting Started

Please refer [here](./INSTALL.md) for installation instructions and dataset preparation.

After installation step is done, you can utilize our model in just a few lines as below. `ZimPredictor` is compatible with `SamPredictor`, such as `set_image()` or `predict()`.
```python
from zim import zim_model_registry, ZimPredictor

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
from zim import zim_model_registry, ZimAutomaticMaskGenerator
from zim.utils import show_mat_anns

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

Moreover, we provide a Gradio demo code in `demo/gradio_demo.py`. You can run our model demo locally by running:

```bash
bash demo/gradio_demo.py
```


## Evaluation

We provide an evaluation script, which includes a comparison with SAM, in `script/run_eval.sh`. Make sure the dataset structure is prepared according to the [instructions](./INSTALL.md).

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

![Table]([./assets/Table1.png](https://github.com/naver-ai/ZIM/releases/download/asset-v1/Table1.png))

## License

```
ZIM
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)  
```
