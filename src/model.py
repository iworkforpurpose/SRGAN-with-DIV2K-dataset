import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ------------------------------
#  Residual Block for Generator
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# ------------------------------
#          Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self, num_residual_blocks: int = 16, upsample_factor: int = 4):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        res_blocks = [ResidualBlock(64) for _ in range(num_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.post_res = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        upsample_layers = []
        for _ in range(int(upsample_factor // 2)):
            upsample_layers += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.res_blocks(x)
        x = self.post_res(x)
        x = x + residual
        x = self.upsample(x)
        return self.final(x)

# ------------------------------
#       Discriminator
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(Discriminator, self).__init__()
        in_channels = input_shape[0]
        def d_block(in_f, out_f, stride=1, bn=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        layers.extend(d_block(in_channels, 64, bn=False))
        layers.extend(d_block(64, 64, stride=2))
        layers.extend(d_block(64, 128))
        layers.extend(d_block(128, 128, stride=2))
        layers.extend(d_block(128, 256))
        layers.extend(d_block(256, 256, stride=2))
        layers.extend(d_block(256, 512))
        layers.extend(d_block(512, 512, stride=2))
        layers.append(nn.AdaptiveAvgPool2d((6,6)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512*6*6, 1024))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(1024, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, img): return self.model(img)

# ------------------------------
#  VGG Feature Extractor for Content Loss
# ------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer: int = 35, use_bn: bool = False, device: str = 'cpu'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19_bn(pretrained=True) if use_bn else models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer]).to(device)
        for p in self.features.parameters(): p.requires_grad = False
    def forward(self, img): return self.features(img)

# ------------------------------
#  Loss Criteria
# ------------------------------
adversarial_criterion = nn.BCEWithLogitsLoss()
content_criterion = nn.MSELoss()