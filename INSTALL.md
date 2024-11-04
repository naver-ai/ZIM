# Installation

We provide installation and dataset preparation instructions here.

## Environment Setup

Install required packages with the command below:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1) MicroMat-3K Dataset
![MicroMat-3K](./assets/qualitative_micromat.png)
In our paper, we introduce a new test set named MicroMat-3K, to evaluate zero-shot interactive matting models. It consists of 3,000 high-resolution images paired with micro-level matte labels, providing a comprehensive benchmark for testing various matting models under different levels of detail.

Downloading MicroMat-3K dataset is available [here](https://huggingface.co/datasets/naver-iv/MicroMat-3K)

### 2) Custom Dataset
You can prepare custom dataset following below instructions.

#### 2-1) Dataset structure

Dataset structure should be as below:
```bash
└── /path/to/dataset/
    ├── img
    │   ├── 0001.png
    ├── matte
    │   ├── coarse
    │   │   ├── 0001.png
    │   └── fine
    │       ├── 0001.png
    ├── prompt
    │   ├── coarse
    │   │   ├── 0001.png
    │   └── fine
    │       ├── 0001.png
    └── seg
        ├── coarse
        │   ├── 0001_01.json
        └── fine
            ├── 0001_01.json
```

#### 2-2) Prompt file configuration

Prompt file configuration should be as below:
```json
{
    "point": [[x1, y1, 1], [x2, y2, 0], ...],   # 1: Positive, 0: Negative prompt
    "bbox": [x1, y1, x2, y2]                    # [X, Y, X, Y] format
}
```