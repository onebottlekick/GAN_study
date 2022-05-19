import glob
import os
import random
import zipfile
from urllib import request

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import DownloadProgress


def to_rgb(img):
    rgb_img = Image.new('RGB', img.size)
    rgb_img.paste(img)
    return rgb_img


class ImageDataset(Dataset):
    def __init__(self, root='datasets', download=False, transform=None, unaligned=False, train=True, dataset_name='horse2zebra'):
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned
        
        if not os.path.exists(root) or download:
            try:
                self._download()
            except Exception as e:
                print(e)
                raise Exception( "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos")


                
        
        if train:
            self.files_A = sorted(glob.glob(os.path.join(root, self.dataset_name, 'trainA', '*.*')))
            self.files_B = sorted(glob.glob(os.path.join(root, self.dataset_name, 'trainB', '*.*')))
            
        else:
            self.files_A = sorted(glob.glob(os.path.join(root, self.dataset_name, 'testA', '*.*')))
            self.files_B = sorted(glob.glob(os.path.join(root, self.dataset_name, 'testB', '*.*')))
        
        if len(self.files_A) == 0:
            raise Exception('Use download=True')
        
    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index%len(self.files_A)])
        
        if self.unaligned:
            img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            img_B = Image.open(self.files_B[index%len(self.files_B)])
        
        if img_A.mode != 'RGB':
            img_A = to_rgb(img_A)
        if img_B.mode != 'RGB':
            img_B = to_rgb(img_B)
            
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {'A':item_A, 'B':item_B}
    
    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        
        URL = f'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{self.dataset_name}.zip'

        with DownloadProgress(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'{self.dataset_name + ".zip"}') as t:
            file_path, _ = request.urlretrieve(URL, reporthook=t.update_to)
            
        with zipfile.ZipFile(file_path) as f:
            for name in tqdm(iterable=f.namelist(), total=len(f.namelist())):
                f.extract(member=name, path=self.root)


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
