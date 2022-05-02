import os
import cv2
import time
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms

from dataset import TestDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from vision_transformer import vit_base
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler

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
    test_data = TestDataset('./open/test')

    # trainning set 중 validation 데이터로 사용할 비율
    valid_size = 0.2

    # validation으로 사용할 trainning indices를 얻는다.
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # train_idx, valid_idx, _, _ = train_test_split(indices, label, test_size=0.2, random_state=1004, shuffle=True, stratify=label)
    
    
    # trainning, validation batch를 얻기 위한 sampler정의
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=batch_size,
    #                                            sampler=train_sampler,
    #                                            num_workers=4)
    train_loader = torch.utils.data.DataLoader(train_data,
                                            sampler=ImbalancedDatasetSampler(train_data),
                                            batch_size=batch_size,
                                            num_workers=4)
    # load validation data in batches
    # valid_loader = torch.utils.data.DataLoader(train_data,
    #                                            batch_size=batch_size,
    #                                            sampler=valid_sampler,
    #                                            num_workers=4)
    valid_loader = None

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return train_data, train_loader, valid_loader, test_loader

def main():
    
    ## mean, std 계산하기
    # data_dir = "train"
    # get_mean_std(data_dir)
    # data_dir = "./"
    # 512 x 512 일 때
    # 평균 0.43336138 0.4037862 0.39447215
    # 표준편차 0.18141639 0.1738924 0.16310532

    ## model initial and load pretrained state
    model = vit_base(patch_size=16, num_classes=88)
    stat_dict = torch.load('/nas/home/tmddnjs3467/dacon/ood_detection/dino_vitbase16_pretrain.pth', map_location='cpu')
    # for key in list(stat_dict['state_dict'].keys()):
    #     stat_dict['state_dict'][key.replace('module.base_encoder.', '')] = stat_dict['state_dict'].pop(key)
    model.load_state_dict(stat_dict, strict=False)
    # import pdb;pdb.set_trace()
    model = torch.nn.DataParallel(model).cuda()

    train_data, train_loader, valid_loader, test_loader = create_datasets(batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    ## finetune
    epochs = 50
    for epoch in range(epochs):
        start = time.time()

        train_loss = 0
        train_pred = []
        train_y = []
        
        val_loss = 0
        val_pred = []
        val_y = []

        model.train()
        
        for batch in (train_loader):
            optimizer.zero_grad()

            x, y = batch[0].cuda(), batch[1].cuda()
            
            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()/len(train_loader)
            train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
            train_y += y.detach().cpu().numpy().tolist() 
    
        train_f1 = score_function(train_y, train_pred)

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
    submission.to_csv("./dino_baseline.csv", index=False)

if __name__ == "__main__":
    main() 
