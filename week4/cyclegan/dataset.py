import glob
import random
import os
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


def to_rgb(img):
    rgb_img = Image.new('RGB', img.size)
    rgb_img.paste(img)
    return rgb_img


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned
        
        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        
    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index%len(self.files_A)])
        
        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index%self.files_B])
        
        if img_A.mode != 'RGB':
            img_A = to_rgb(img_A)
        if img_B.mode != 'RGB':
            img_B = to_rgb(img_B)
            
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {'A':item_A, 'B':item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))