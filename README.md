# ZIM

**ZIM: Zero-Shot Image Matting for Anything** <br />
[Beomyoung Kim](https://beomyoung-kim.github.io/), Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, Joonsang Yu<br>

<sub>NAVER Cloud, ImageVision</sub><br />

[![Paper](https://img.shields.io/badge/Paper-arxiv)](https://arxiv.org) 	
[![Page](https://img.shields.io/badge/Project_page-blue)](https://naver-ai.github.io/ZIM) 	
[![Demo](https://img.shields.io/badge/Demo-yellow)](https://huggingface.co/spaces/naver-iv/ZIM_Zero-Shot-Image-Matting)
[![Data](https://img.shields.io/badge/Data-gray)](https://huggingface.co/datasets/naver-iv/MicroMat-3K)



## Introduction

In this paper, we introduce a novel zero-shot image matting model. Recent models like SAM (Segment Anything Model) exhibit strong zero-shot capabilities, but they fall short in generating fine-grained, high-precision masks. To address this limitation, we propose two key contributions: First, we develop a label converter that transforms segmentation labels into detailed matte labels, creating the new SA1B-Matte dataset. This enables the model to generate high-quality, micro-level matte masks without costly manual annotations. Second, we design a zero-shot matting model equipped with a hierarchical pixel decoder and prompt-aware masked attention mechanism, improving both the resolution of mask outputs and the model’s ability to focus on specific regions based on user prompts. We evaluate our model using the newly introduced ZIM test set, which contains high-quality micro-level matte labels. Experimental results show that our model outperforms SAM and other existing methods in precision and zero-shot generalization. Furthermore, we demonstrate the versatility of our approach in downstream tasks, including image inpainting and 3D neural radiance fields (NeRF), where the ability to produce precise matte masks is crucial. Our contributions provide a robust foundation for advancing zero-shot image matting and its applications across a wide range of computer vision tasks.


## Updates    
**Available Soon**


## Installation

Our implementation is based on [SAM](https://github.com/facebookresearch/segment-anything).

Please check the [installation instructions](INSTALL.md)

## License

Available Soon