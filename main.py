import os
import cv2
import time
import tqdm
import torch
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
ap.add_argument("--mode", required = True, type = str, help='IBOT or MOCO or DINO or BASE')
ap.add_argument("--epochs", default = 25, type = int, help='number of training epochs')
ap.add_argument("--batch_size", default = 32, type = int, help='training batchsize')
# ap.add_argument("--save_path", required = True, type = str, help='path to storage the loss table, stat_dict, csv')
ap.add_argument("--lr", default = 5e-5, type = float, help='learning rate , default 5e-5')
ap.add_argument("--wd", default = 5e-5, type = float, help='weight decay, default 5e-3')
ap.add_argument("--SAM", default = False, type = bool, help='using SAM')
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
    # transforms.RandomHorizontalFlip(),  # 좌우반전 
    # transforms.RandomVerticalFlip(),  # 상하반전 
    transforms.Resize((512, 512)),  # 알맞게 변경하세요 
    transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
    transforms.Normalize([0.43336138, 0.4037862, 0.39447215], [0.18141639, 0.1738924, 0.16310532])  #  정규화(normalization)
    ])
    

    # choose the training and test datasets
    train_data = datasets.ImageFolder(os.path.join('./train'), train_transform)
    label = [i[1] for i in train_data.samples]
    # import pdb;pdb.set_trace()
    test_data = TestDataset('./open/test')

    # trainning set 중 validation 데이터로 사용할 비율
    # valid_size = 0.2

    # validation으로 사용할 trainning indices를 얻는다.
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # train_idx, valid_idx, _, _ = train_test_split(indices, label, test_size=0.2, random_state=1004, shuffle=True, stratify=label)
    
    
    # trainning, validation batch를 얻기 위한 sampler정의
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                         sampler=ImbalancedDatasetSampler(train_data),
    #                                         batch_size=batch_size,
    #                                         num_workers=4)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=4)
    # valid_loader = None

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return train_data, train_loader, valid_loader, test_loader

def main():
    
    ## mean, std 계산하기
    # data_dir = "train"
    # get_mean_std(data_dir)
    # data_dir = "./"
    # import pdb;pdb.set_trace()
    # 512 x 512 일 때
    # 평균 0.43336138 0.4037862 0.39447215
    # 표준편차 0.18141639 0.1738924 0.16310532
    # 224 x 224 일 때
    # 평균 0.4330423 0.40345937 0.3941526
    # 표준편차 0.177745 0.1705442 0.16002317

    ## save path
    args['save_path'] = './exp/' + args['mode'] + '_lr' + str(args['lr']) + '_wd' + str(args['wd']) + '_e' + str(args['epochs'])
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
    # for key in list(stat_dict['state_dict'].keys()):
    #     stat_dict['state_dict'][key.replace('module.small_encoder.', '')] = stat_dict['state_dict'].pop(key)
    # import pdb;pdb.set_trace()
    model.load_state_dict(stat_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()

    train_data, train_loader, valid_loader, test_loader = create_datasets(batch_size=args['batch_size'])

    if args['SAM']:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=args['lr'], weight_decay=args['wd'])
    else: optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    criterion = nn.CrossEntropyLoss().cuda()

    ## finetune
    train_loss_list = []
    train_f1_list = []
    epochs = args['epochs']
    for epoch in range(epochs):
        start = time.time()

        train_loss = 0
        train_pred = []
        train_y = []
        
        # val_loss = 0
        # val_pred = []
        # val_y = []

        model.train()
        
        for batch in (train_loader):
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
        # model.eval()
        # with torch.no_grad():
        #     for batch in (valid_loader):
        #         x, y = batch[0].cuda(), batch[1].cuda()
                
        #         pred = model(x)
        #         loss = criterion(pred, y)
                
        #         val_loss += loss.item()/len(valid_loader)
        #         val_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        #         val_y += y.detach().cpu().numpy().tolist() 
        
        # val_f1 = score_function(val_y, val_pred)
        
        TIME = time.time() - start
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
        # print(f'VALID    loss : {val_loss:.5f}    f1 : {val_f1:.5f}')
    with open(args['save_path'] + '/model.pkl', 'wb') as f:
        pickle.dump(model.state_dict(), f)
    with open(args['save_path'] + '/loss_f1.pkl', 'wb') as f:
        pickle.dump([train_loss_list, train_f1_list], f)
    
    ## test
    model.eval()
    f_pred = []
    with torch.no_grad():
        for batch in (test_loader):
            x = batch.cuda()
            pred = model(x)
            f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())
    
    # import pdb;pdb.set_trace()
    idx_to_class = {value:key for key,value in train_data.class_to_idx.items()}
    f_result = [idx_to_class[result] for result in f_pred]

    submission = pd.read_csv("open/sample_submission.csv")
    submission["label"] = f_result
    submission.to_csv(args['save_path'] + '/test.csv', index=False)

if __name__ == "__main__":
    main() 
