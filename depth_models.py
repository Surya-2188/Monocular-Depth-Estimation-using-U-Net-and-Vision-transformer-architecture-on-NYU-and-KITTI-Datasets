import math
import torch
from torch import nn
from torch.nn import functional as F

HEIGHT, WIDTH = 256, 512


class ViTDepth(nn.Module):
    def __init__(self, height=HEIGHT, width=WIDTH):
        super().__init__()
        patch_size, dim = 16, 128
        self.grid = (height // patch_size, width // patch_size)
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.empty(1, math.prod(self.grid), dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=4 * dim, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=4, norm=nn.LayerNorm(dim), enable_nested_tensor=False,
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, rgb):
        patches = self.patch_embedding(rgb)
        tokens = patches.flatten(2).transpose(1, 2)
        features = self.encoder(tokens + self.position).transpose(1, 2)
        features = features.reshape(rgb.shape[0], -1, *self.grid)
        depth = self.depth_head(features)
        return F.softplus(F.interpolate(
            depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False,
        ))


class UNetDepth(nn.Module):
    def __init__(self):
        super().__init__()

        def block(inputs, outputs):
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(),
            )

        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.bridge = block(64, 128)
        self.dec3 = block(128 + 64, 64)
        self.dec2 = block(64 + 32, 32)
        self.dec1 = block(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, rgb):
        a = self.enc1(rgb)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        value = self.bridge(self.pool(c))
        for skip, decode in ((c, self.dec3), (b, self.dec2), (a, self.dec1)):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decode(torch.cat((value, skip), dim=1))
        return F.softplus(self.head(value))
