#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn

from model.squeeze_and_excitation_3D import ChannelSELayer3D


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16, use_prelu=False, se_at_end=False,
                 se_throughout=False):
        super().__init__()
        self.first = ConvBlock(in_channels, first_out_channels, use_prelu)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels, use_prelu)
        self.down2 = Down(2 * in_channels, 4 * in_channels, use_prelu)
        self.down3 = Down(4 * in_channels, 8 * in_channels, use_prelu)
        self.up1 = Up(8 * in_channels, 4 * in_channels, use_prelu)
        self.se1 = ChannelSELayer3D(4 * in_channels, reduction_ratio=16) if se_throughout else None
        self.up2 = Up(4 * in_channels, 2 * in_channels, use_prelu)
        self.se2 = ChannelSELayer3D(2 * in_channels, reduction_ratio=16) if se_throughout else None
        self.up3 = Up(2 * in_channels, in_channels, use_prelu)
        self.se3 = ChannelSELayer3D(in_channels, reduction_ratio=16) if se_at_end else None
        self.final = nn.Conv3d(in_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        if self.se1 is not None:
            x = self.se1(x)
        x = self.up2(x, x2)
        if self.se2 is not None:
            x = self.se2(x)
        x = self.up3(x, x1)
        if self.se3 is not None:
            x = self.se3(x)
        x = self.final(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_prelu=False):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU() if use_prelu else nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU() if use_prelu else nn.LeakyReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_prelu=False):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, use_prelu)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_prelu=False):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, use_prelu)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.PReLU() if use_prelu else nn.LeakyReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x


if __name__ == '__main__':
    x = torch.rand((1, 1, 64, 64, 64))
    model = UNet(1, 1, use_prelu=True, se_at_end=True, se_throughout=True)
    print(x.shape)
    print(model)
    result = model(x)
    print(result.shape)
