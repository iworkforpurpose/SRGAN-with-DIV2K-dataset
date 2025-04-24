import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------
# Dense Block (used inside RRDB)
# ----------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_channels=32):
        super(DenseBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        c1 = self.lrelu(self.conv1(x))
        c2 = self.lrelu(self.conv2(torch.cat([x, c1], 1)))
        c3 = self.lrelu(self.conv3(torch.cat([x, c1, c2], 1)))
        c4 = self.lrelu(self.conv4(torch.cat([x, c1, c2, c3], 1)))
        c5 = self.conv5(torch.cat([x, c1, c2, c3, c4], 1))
        return x + 0.2 * c5


# ----------------------------
# RRDB Block (Residual in Residual Dense Block)
# ----------------------------
class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.block = nn.Sequential(
            DenseBlock(channels),
            DenseBlock(channels),
            DenseBlock(channels)
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)


# ----------------------------
# ESRGAN Generator
# ----------------------------
class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=23, scale=4):
        super(GeneratorRRDB, self).__init__()
        self.initial = nn.Conv2d(in_channels, channels, 3, 1, 1)

        # Trunk: 23 RRDB blocks
        self.trunk = nn.Sequential(*[RRDB(channels) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(channels, channels, 3, 1, 1)

        # Upsampling
        upsample_layers = []
        for _ in range(int(scale).bit_length() - 1):
            upsample_layers += [
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.final = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.initial(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk
        out = self.upsample(fea)
        return self.final(out)


# ----------------------------
# ESRGAN Discriminator
# ----------------------------
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


# ----------------------------
# VGG Feature Extractor (for perceptual loss)
# ----------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer=34, use_bn=False, device='cpu'):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:layer]).to(device)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, img):
        return self.features(img)
