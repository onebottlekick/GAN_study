import torch
from torch import nn
from math import log2


class WeightScaledConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WeightScaledConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2/(in_channels*kernel_size**2))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return self.conv(x*self.scale) + self.bias.view(self.bias.shape[0], 1, 1)
    
class WeightScaledConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=0):
        super(WeightScaledConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2/(in_channels*kernel_size**2))**0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        return self.conv(x*self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x/torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pexelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pixelnorm = use_pexelnorm
        self.conv1 = WeightScaledConv2d(in_channels, out_channels)
        self.conv2 = WeightScaledConv2d(out_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.pixel_norm(x) if self.use_pixelnorm else x
        x = self.leaky_relu(self.conv2(x))
        x = self.pixel_norm(x) if self.use_pixelnorm else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3, factors=[1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            WeightScaledConvTranspose2d(z_dim, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WeightScaledConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        
        self.to_rgb = WeightScaledConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.to_rgb])        
        
        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels*factors[i])
            conv_out_channels = int(in_channels*factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WeightScaledConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha*generated + (1 - alpha)*upscaled)
        
    def forward(self, x, alpha, steps):
        out = self.initial(x)
        
        if steps == 0:
            return self.to_rgb(out)
        
        for step in range(steps):
            upscaled = nn.Upsample(scale_factor=2, mode='nearest')(out)
            out = self.prog_blocks[step](upscaled)
        
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Critic(nn.Module):
    def __init__(self, in_channels, img_channels=3, factors=[1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]):
        super(Critic, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels*factors[i])
            conv_out_channels = int(in_channels*factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pexelnorm=False))
            self.rgb_layers.append(WeightScaledConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))
        
        self.to_rgb = WeightScaledConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.to_rgb)
        self.avg_pool = nn.AvgPool2d(2, 2)
        
        self.final_block = nn.Sequential(
            WeightScaledConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def fade_in(self, alpha, downscaled, out):
        return alpha*out + (1 - alpha)*downscaled
        
    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)
        
    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        
        out = self.leaky_relu(self.rgb_layers[cur_step](x))
        
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.leaky_relu(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        
        for step in range(cur_step+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
            
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)