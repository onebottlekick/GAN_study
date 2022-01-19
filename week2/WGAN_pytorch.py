import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim, img_channel, generator_features):
        super(Generator, self).__init__()
        
        def _block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        
        self.model = nn.Sequential(
            _block(z_dim, generator_features*8, kernel_size=4, stride=1, padding=0),
            _block(generator_features*8, generator_features*4, kernel_size=4, stride=2, padding=1),
            _block(generator_features*4, generator_features*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(generator_features*2, img_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)
    
    
class Critic(nn.Module):
    def __init__(self, img_channel, critic_features):
        super(Critic, self).__init__()
        
        def _block(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2)
            )
            
        self.model = nn.Sequential(
            nn.Conv2d(img_channel, critic_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            _block(critic_features, critic_features*2, kernel_size=4, stride=2, padding=1),
            _block(critic_features*2, critic_features*4, kernel_size=4, stride=2, padding=1),
            _block(critic_features*4, critic_features*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(critic_features*8, 1, kernel_size=4, stride=2, padding=1)
        )
            
    def forward(self, img):
        return self.model(img)