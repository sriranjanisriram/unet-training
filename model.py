# Author: Sriranjani Sriram

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double Convolution block with Batch Normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        """Initializes the DoubleConv block."""

        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass through the block."""

        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net model for image segmentation."""

    def __init__(self, in_channels, out_channels):
        """Initializes the U-Net model."""

        super(UNet, self).__init__()
        self.down_conv1 = DoubleConv(in_channels, 8)
        self.down_conv2 = DoubleConv(8, 16)
        self.down_conv3 = DoubleConv(16, 32)
        self.down_conv4 = DoubleConv(32, 64)
        self.down_conv5 = DoubleConv(64, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.up_conv4 = DoubleConv(192, 64)
        self.up_conv3 = DoubleConv(96, 32)
        self.up_conv2 = DoubleConv(48, 16)
        self.up_conv1 = DoubleConv(24, 8)

        self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the U-Net model."""

        # Downward path
        x1 = self.down_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.down_conv5(x8)

        # Upward path
        x = self.upsample(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.up_conv4(x)
        x = self.upsample(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv3(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv1(x)

        # Output
        x = self.out_conv(x)
        return x