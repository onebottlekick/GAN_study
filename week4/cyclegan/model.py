import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features)
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()
        
        channels = input_shape[0]
        
        out_features = 64
        
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU()
        ]
        
        in_features = out_features
        
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
            
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        channels, height, width = input_shape
        
        self.output_shape = (1, height//(2**4), width//(2**4))
        
        self.model = nn.Sequential(
            *self._block(channels, 64, normalize=False),
            *self._block(64, 128),
            *self._block(128, 256),
            *self._block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def _block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return layers
    
    def forward(self, img):
        return self.model(img)