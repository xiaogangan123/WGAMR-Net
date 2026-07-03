import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, c1, stride=1),
            ConvBlock(c1, c1, stride=1),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(c1, c2, stride=2),
            ConvBlock(c2, c2, stride=1),
        )
        self.enc3 = nn.Sequential(
            ConvBlock(c2, c3, stride=2),
            ConvBlock(c3, c3, stride=1),
        )
        self.enc4 = nn.Sequential(
            ConvBlock(c3, c4, stride=2),
            ConvBlock(c4, c4, stride=1),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec3 = nn.Sequential(
            ConvBlock(c4 + c3, c3, stride=1),
            ConvBlock(c3, c3, stride=1),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec2 = nn.Sequential(
            ConvBlock(c3 + c2, c2, stride=1),
            ConvBlock(c2, c2, stride=1),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec1 = nn.Sequential(
            ConvBlock(c2 + c1, c1, stride=1),
            ConvBlock(c1, c1, stride=1),
        )

        self.flow_head = nn.Conv3d(c1, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.flow_head(d1)


class VoxelMorph(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        self.unet = UNet3D(in_channels=in_channels, base_channels=base_channels)

    def forward(self, x):
        return self.unet(x)



