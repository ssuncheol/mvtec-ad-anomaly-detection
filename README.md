# Team Project in Deep Learning 

This project conducted a comparative experiment on the anomaly detection performance between several models. [ViT(supervised), Moco v3, DINO, iBOT]


## Requirements 

```shell
Cuda >= 10.2
Python3 >= 3.7
PyTorch >= 1.8.0
Torchvision >= 0.10.0
Imgaug # 
``` 

# Quickstart 

## Cloning a repository 

```shell
git clone https://github.com/tjtmddnjswkd/DL_TEAM_PROJECT.git 
```

## Prepare a dataset

Our team used the MVTec AD dataset provided by [MVTec AD Research](https://www.mvtec.com/company/research/datasets/mvtec-ad). The MVTec AD Dataset consists of n trains and m validation sets. In the case of the above task, there are 15 tasks to be classified.


## Data augmentation 

We experimented with different augmentation for each class object following the paper below. It was implemented on "augmentation.py"

<img width="606" alt="스크린샷 2022-06-02 오후 5 13 47" src="https://user-images.githubusercontent.com/52492949/171585717-8631dc48-9439-4ab7-9659-e3e027820d19.png">


## Model 

Our team used ViT[1], Moco[2], DINO[3], iBOT[4]. 

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

First, a hyper-parameter setting suitable for each model was found through grid search algorithm. In this project, after finding the optimal hyper-parameter for each model, experiments were conducted based on various conditions. The experiment proceeded as follows. 

```shell 
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --mode BASE --lr 5e-5 --wd 1e-4 --FL True & 
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --mode MOCO --lr 1e-4 --wd 1e-2 --FL True &
CUDA_VISIBLE_DEVICES=4,5 python3 main.py --mode DINO --lr 5e-6 --wd 1e-1 --FL True &
CUDA_VISIBLE_DEVICES=6,7 python3 main.py --mode IBOT --lr 1e-5 --wd 1e-1 --FL True 
```

## Result 

Baseline 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 | 5e-5 	| 1e-4 | 69.7 |
| Moco 	| MVTec-AD  	| 32 | 1e-4 	| 1e-2 | 45.74|
| DINO	| MVTec-AD  	| 32 | 5e-6 	| 1e-1| 60.95|
| iBOT 	| MVTec-AD  	| 32 | 1e-5 	| 1e-1 | 63.86|

Baseline + Augmentation 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 | 5e-5 	| 1e-4 | 77.69 |
| Moco 	| MVTec-AD  	| 32 |  1e-4	| 1e-2 | 44.9 |
| DINO	| MVTec-AD  	| 32 | 5e-6 	| 1e-1| 71.8|
| iBOT 	| MVTec-AD  	| 32 |  1e-5 	| 1e-1 | 73.25|

Baseline + Augmentation + Add the train data(outlier) 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  5e-5	| 1e-4| 79.97 |
| Moco 	| MVTec-AD  	| 32 |  1e-4	| 1e-2 | 74.63 |
| DINO	| MVTec-AD  	| 32 | 5e-6 	| 1e-1| 75.8|
| iBOT 	| MVTec-AD  	| 32 | 1e-5 	| 1e-1 | 80.41|

Baseline + Augmentation + Add the train data(outlier) + Focal loss[5] 

| Model 	| Dataset 	| Batch size | Learning rate | Weight decay | Accuracy(test) 	|
|:---------:|:--------:|:---------------------------------------:|:-----:|:---:|:---:|
| Base(ViT) 	| MVTec-AD 	| 32 |  	  5e-5	| 1e-4 | 80.13 |
| Moco 	| MVTec-AD  	| 32 |  	1e-4	| 1e-2 | 73.59 |
| DINO	| MVTec-AD  	| 32 |  	5e-6 	| 1e-1| 77.12|
| iBOT 	| MVTec-AD  	| 32 |  	1e-5 	| 1e-1 | 80.31|





### Reference 

- [1] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)   
- [2] [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)  
- [3] [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)  
- [4] [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
- [5] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)




















