# Team Project in Deep Learning 

This project conducted a comparative experiment on the anomaly detection performance between several models. [ViT, Moco, DINO, IBOT]


## Requirements 

```shell
Cuda >= 10.2
Python3 >= 3.7
PyTorch >= 1.8.0
Torchvision >= 0.10.0
Imgaug # 
``` 

## Quickstart 

### Cloning a repository 

```shell
git clone https://github.com/tjtmddnjswkd/DL_TEAM_PROJECT.git 
```

### Prepare a dataset(MVTec AD Dataset)

Our team used the MVTec AD dataset provided by [MVTec AD Research](https://www.mvtec.com/company/research/datasets/mvtec-ad, "mvtec-ad"). The MVTec AD Dataset consists of n trains and m validation sets. 

### Model 

Our team using ViT, Moco, DINO, IBOT. 

### Usage 

It is implemented under multi-gpu setting using data parallel. 

### Arguments 

| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Mode 	| [str] 	| IBOT, Moco, DINO, BASE | BASE|
| Epochs 	| [int] 	| Number of training epochs| 25|
| Batch size | [int]	| Training batch size| 32|
| Learning rate 	| [float]	| Learning rate| 	5e-5|
| Weight_decay 	| [float]	| Weight decay | 5e-3|
| FL 	| [bool]	| Using focal loss | False|

### How to train 

```shell 

```

### Result 

1. Baseline 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  	| | |
| Moco 	| MVTec-AD  	| 32 |  	| | |
| DINO	| MVTec-AD  	| 32 |  	| | |
| IBOT 	| MVTec-AD  	| 32 |  	| | |

2. Baseline with augmentation 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  	| | |
| Moco 	| MVTec-AD  	| 32 |  	| | |
| DINO	| MVTec-AD  	| 32 |  	| | |
| IBOT 	| MVTec-AD  	| 32 |  	| | |

3. Baseline with train augmentation 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  	| | |
| Moco 	| MVTec-AD  	| 32 |  	| | |
| DINO	| MVTec-AD  	| 32 |  	| | |
| IBOT 	| MVTec-AD  	| 32 |  	| | |

4. Baseline with focal loss 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  	| | |
| Moco 	| MVTec-AD  	| 32 |  	| | |
| DINO	| MVTec-AD  	| 32 |  	| | |
| IBOT 	| MVTec-AD  	| 32 |  	| | |





### Reference 

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929, "Vision Transformer")
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722, "Moco")
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294, 'DINO')
- [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832, "IBOT")
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002, "Focal loss")




















