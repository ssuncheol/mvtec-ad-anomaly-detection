# Team Project in Deep Learning 

This project conducted a comparative experiment on the anomaly detection performance between several models. [ViT, Moco, DINO, IBOT]


## Requirements 

```shell
Cuda 11.0
Python3 3.8
PyTorch 1.8
Torchvision 0.10.0
Imgaug # 
``` 

## Quickstart 

### Cloning a repository 

```shell
git clone https://github.com/tjtmddnjswkd/DL_TEAM_PROJECT.git 
```

### Prepare a dataset(MVTec AD Dataset)

Our team used the MVTec AD dataset provided by https://www.mvtec.com/company/research/datasets/mvtec-ad. The MVTec AD Dataset consists of n trains and m validation sets. 

### Model 

Our team use ViT, Moco, DINO, IBOT. 

## Experiments 

### Arguments 

| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Epochs 	| [int] 	| Epochs | 300|
| Batch_size 	| [int] 	| Batch size| 1024|
| Model 	| [str]	| Vision Transformer| 	vit|
| Optimizer 	| [str]	| Adam, Adamw| 	AdamW|
| Weight_decay 	| [float]	| Weight decay | 0.3|
| T_max 	| [int]	| Cosine Annealing step | 80000 |
| Dropout 	| [float]	| Dropout | 0.0|
| World_size 	| [int]	| World size | 8 |


### How to train 


### How to eval 


### Result 


### Reference 

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929, "Vision Transformer")
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722, "Moco")
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294, 'DINO')
- [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832, "IBOT")
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002, "Focal loss")




















