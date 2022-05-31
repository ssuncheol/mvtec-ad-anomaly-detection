import os
import cv2
import time
import tqdm
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import pickle

from dataset import TestDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from vision_transformer import vit_small
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, type=str, help='IBOT or MOCO or DINO or BASE')
ap.add_argument("--epochs", default=25, type=int, help='number of training epochs')
ap.add_argument("--batch_size", default=32, type=int, help='training batchsize')
# ap.add_argument("--save_path", required=True, type=str, help='path to storage the loss table, stat_dict, csv')
ap.add_argument("--lr", default=5e-5, type=float, help='learning rate , default 5e-5')
ap.add_argument("--wd", default=1e-2, type=float, help='weight decay, default 5e-3')
ap.add_argument("--SAM", default=False, type=bool, help='using SAM')
ap.add_argument("--FL", default=False, type=bool, help='using focal loss')
args = vars(ap.parse_args())

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def get_mean_std(data_dir):
    '''
    이미지 정규화 시 성능 향상 , 평균과 표준편차로 정규화 실행
    data_dir = 이미지 들어있는 폴더 path
    '''
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(os.path.join(f'./{data_dir}'), transform)
    print("데이터 정보", dataset)

    meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    print("평균",meanR, meanG, meanB)
    print("표준편차",stdR, stdG, stdB)

def create_datasets(batch_size):
    
    train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
    transforms.Normalize([0.43322375, 0.40371937, 0.39443374], [0.18148199, 0.17403568, 0.16333582])  #  정규화(normalization)
    ])
    
    # choose the training and test datasets
    train_data = datasets.ImageFolder(os.path.join('/data/mvtec_ad_train'), train_transform)
    test_data = datasets.ImageFolder(os.path.join('/data/mvtec_ad_test'), train_transform)
    label = [i[1] for i in train_data.samples]
    
    # trainning set 중 validation 데이터로 사용할 비율
    valid_size = 0.2

    # validation으로 사용할 trainning indices를 얻는다.
    num_train = len(train_data)
    indices = list(range(num_train))
    train_idx, valid_idx, _, _ = train_test_split(indices, label, test_size=0.2, random_state=42, shuffle=True, stratify=label)
    
    
    # trainning, validation batch를 얻기 위한 sampler정의
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=4)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=4)
    
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              shuffle=False, 
                                              batch_size=batch_size)

    return train_data, train_loader, valid_loader, test_loader

def main():
    
    ## mean, std 계산하기
    # data_dir = "./train_images"
    # get_mean_std(data_dir)
    # 512 x 512 일 때
    # 평균 0.43322375 0.40371937 0.39443374
    # 표준편차 0.18148199 0.17403568 0.16333582

    ## set seed
    seed_torch(42)
    
    ## save path
    args['save_path'] = './exp/' + args['mode'] + '_lr' + str(args['lr']) + '_wd' + str(args['wd']) + '_e' + str(args['epochs'])
    if args['SAM']:
        args['save_path'] += 'SAM'
    if not os.path.exists(args['save_path']):
        os.mkdir(args['save_path'])
    with open(args['save_path'] + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    ## pretrained model path
    if args['mode'] == 'MOCO':
        model_path = '/nas/home/tmddnjs3467/dacon/ood_detection/small16/moco_small16_pretrain.pth'
    elif args['mode'] == 'DINO':
        model_path = '/nas/home/tmddnjs3467/dacon/ood_detection/small16/dino_small16_pretrain.pth'
    elif args['mode'] == 'IBOT':
        model_path = '/nas/home/tmddnjs3467/dacon/ood_detection/small16/ibot_small_16_pretrain.pth'
    else: model_path = '/nas/home/tmddnjs3467/dacon/ood_detection/small16/vit_small16_sup.pth'

    ## model initial and load pretrained state
    model = vit_small(patch_size=16, num_classes=88)
    stat_dict = torch.load(model_path, map_location='cpu')
    if args['mode'] == 'BASE':
        stat_dict.pop('head.weight')
        stat_dict.pop('head.bias')
    elif args['mode'] == 'IBOT':
        stat_dict = stat_dict['state_dict']
    model.load_state_dict(stat_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()

    train_data, train_loader, valid_loader, test_loader = create_datasets(batch_size=args['batch_size'])

    if args['SAM']:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=args['lr'], weight_decay=args['wd'])
    else: optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    
    if args['FL']:
        criterion = FocalLoss(alpha=0.25, gamma=5.0, label_smoothing=0.1)
    else: criterion = nn.CrossEntropyLoss().cuda()

    ## finetune
    train_loss_list = []
    train_f1_list = []
    val_loss_list = []
    val_f1_list = []

    epochs = args['epochs']
    for epoch in range(epochs):
        start = time.time()

        train_loss = 0
        train_pred = []
        train_y = []
        
        val_loss = 0
        val_pred = []
        val_y = []

        model.train()
        
        for batch in tqdm.tqdm(train_loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            
            pred = model(x)
            loss = criterion(pred, y)

            if args['SAM']:
                loss.backward()
                optimizer.first_step(zero_grad=True)
                criterion(model(x), y).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()/len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += y.detach().cpu().numpy().tolist() 
    
        train_f1 = score_function(train_y, train_pred)
        
        train_loss_list.append(train_loss)
        train_f1_list.append(train_f1)
        
        model.eval()
        with torch.no_grad():
            for batch in (valid_loader):
                x, y = batch[0].cuda(), batch[1].cuda()
                
                pred = model(x)
                loss = criterion(pred, y)
                
                val_loss += loss.item()/len(valid_loader)
                val_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                val_y += y.detach().cpu().numpy().tolist() 
        
        val_f1 = score_function(val_y, val_pred)
        
        val_loss_list.append(val_loss)
        val_f1_list.append(val_f1)

        TIME = time.time() - start
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
        print(f'VALID    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
    
    with open(args['save_path'] + '/model.pkl', 'wb') as f:
        pickle.dump(model.state_dict(), f)
    with open(args['save_path'] + '/train_loss_f1.pkl', 'wb') as f:
        pickle.dump([train_loss_list, train_f1_list], f)
    with open(args['save_path'] + '/val_loss_f1.pkl', 'wb') as f:
        pickle.dump([val_loss_list, val_f1_list], f)
    
    ## test
    model.eval()
    
    test_loss = 0
    test_pred = []
    test_y = []
    
    with torch.no_grad():
        for batch in (test_loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            
            pred = model(x)
            loss = criterion(pred, y)
            
            test_loss += loss.item()/len(test_loader)
            test_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            test_y += y.detach().cpu().numpy().tolist() 
        
        test_f1 = score_function(test_y, test_pred)
    print(f'TEST    loss : {test_loss:.5f}    f1 : {test_f1:.5f}')
    # idx_to_class = {value:key for key,value in train_data.class_to_idx.items()}
    # test_result = [idx_to_class[result] for result in test_pred]
    with open(args['save_path'] + '/test_loss_f1.pkl', 'wb') as f:
        pickle.dump([test_loss, test_f1], f)
    
if __name__ == "__main__":
    main() 
