import torch
from torch import nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_embed = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            *self._block(latent_dim+num_classes, 128, normalize=False),
            *self._block(128, 256),
            *self._block(256, 512),
            *self._block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        
    def _block(self, in_features, out_features, normalize=True):
        layers = [nn.Linear(in_features, out_features)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_features, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        return layers
    
    def forward(self, noise, labels):
        gen_input = torch.cat([self.label_embed(labels), noise], dim=-1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        
        self.label_embed = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(num_classes+int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        
    def forward(self, img, labels):
        d_in = torch.cat([img.view(img.size(0), -1), self.label_embed(labels)], dim=-1)
        validity = self.model(d_in)
        return validity