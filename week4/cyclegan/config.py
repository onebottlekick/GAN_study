import os
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import *


img_channels=3
img_height=256
img_width=256
input_shape = (img_channels, img_height, img_width)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_interval=1
epoch=0
n_epochs=200

batch_size=1

n_residual_blocks=9

lr=0.0002
b1=0.5
b2=0.999
decay_epoch=100

lambda_cyc=10.0
lambda_id=5.0

sample_interval=100

dataset_name="horse2zebra"
transform = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
]
dataloader = DataLoader(
    ImageDataset("../../datasets/%s" % dataset_name, transform=transform, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    ImageDataset("../../datasets/%s" % dataset_name, transform=transform, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
)

os.makedirs("images/%s" % dataset_name, exist_ok=True)
os.makedirs("models/%s" % dataset_name, exist_ok=True)