import cv2
import glob
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = transforms.Compose([   
                        transforms.ToPILImage(),
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.43336138, 0.4037862, 0.39447215], [0.18141639, 0.1738924, 0.16310532])])
        test_png = sorted(glob.glob(f'{self.img_paths}/*.png'))
        self.test_imgs = [img_load(n) for n in tqdm.tqdm(test_png)]

    def __len__(self):
        return len(self.test_imgs)

    def __getitem__(self, idx):
        img = self.transform(self.test_imgs[idx])
        return img
