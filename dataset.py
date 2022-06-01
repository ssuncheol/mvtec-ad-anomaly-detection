import os
import tqdm
import torchvision.transforms as transforms

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from augmentation import *

class MvtecADDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.pre_trans = transforms.Compose([
            transforms.Resize((512, 512))])
                                
        self.post_trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.43322375, 0.40371937, 0.39443374], [0.18148199, 0.17403568, 0.16333582])])
            
        # choose the training and test datasets
        if is_train:
            self.dataset = datasets.ImageFolder(os.path.join('/data/mvtec_ad_train'), self.pre_trans)
            self.label = [i[1] for i in self.dataset.samples] # 0~88
            self.idx_to_class = {value:key for key,value in self.dataset.class_to_idx.items()} # {0: bottle_broken_large, ...}
            self.train_idx, self.valid_idx = self.get_train_val_index()
            
        else: 
            self.dataset = datasets.ImageFolder(os.path.join('/data/mvtec_ad_test'), self.pre_trans)

    def get_train_val_index(self):
        indices = list(range(len(self.dataset.samples)))
        train_idx, valid_idx, _, _ = train_test_split(indices, self.label, test_size=0.2, random_state=42, shuffle=True, stratify=self.label)
        return train_idx, valid_idx
    
    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        label = self.dataset[idx][1]

        if self.is_train:
            if self.idx_to_class[label].split('_')[0] in ['carpet', 'tile', 'leather', 'grid']:
                data = policy1(data)
            elif self.idx_to_class[label].split('_')[0] in ['capsule', 'pill']:
                data = policy2(data)
            elif self.idx_to_class[label].split('_')[0] == 'cable':
                data = policy3(data)
            elif self.idx_to_class[label].split('_')[0] == 'transistor':
                data = policy4(data)
            elif self.idx_to_class[label].split('_')[0] == 'metalnut':
                data = policy5(data)
            elif self.idx_to_class[label].split('_')[0] == 'toothbrush':
                data = policy6(data)
            elif self.idx_to_class[label].split('_')[0] == 'screw':
                data = policy7(data)
            elif self.idx_to_class[label].split('_')[0] == 'hazelnut':
                data = policy8(data)
            elif self.idx_to_class[label].split('_')[0] == 'zipper':
                data = policy9(data)
            elif self.idx_to_class[label].split('_')[0] == 'bottle':
                data = policy10(data)
        
        data = self.post_trans(data)
        return (data, label)

