import torch
import torch.nn as nn
import torchvision.models as models


# Dense block inside RRDB
class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        inputs = x
        for layer in self.block:
            out = layer(inputs)
            inputs = torch.cat([inputs, out], 1) if isinstance(layer, nn.Conv2d) else out
        return inputs * 0.2 + x


# Residual in Residual Dense Block
class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.block = nn.Sequential(
            DenseBlock(channels),
            DenseBlock(channels),
            DenseBlock(channels)
        )

    def forward(self, x):
        return x + self.block(x) * 0.2


# ESRGAN Generator
class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=23):
        super(GeneratorRRDB, self).__init__()
        self.initial = nn.Conv2d(in_channels, channels, 3, 1, 1)

        # 23 RRDB blocks
        self.trunk = nn.Sequential(*[RRDB(channels) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(channels, channels, 3, 1, 1)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final conv
        self.final = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.initial(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        out = self.final(out)
        return out


# Reuse existing discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super(Discriminator, self).__init__()
        in_channels = input_shape[0]

        def d_block(in_f, out_f, stride=1, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f))
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
        layers.append(nn.AdaptiveAvgPool2d((6, 6)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512 * 6 * 6, 1024))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(1024, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# VGG feature extractor (before activation)
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer=34, use_bn=False, device='cpu'):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT) if use_bn else models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer]).to(device)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, img):
        return self.features(img)
