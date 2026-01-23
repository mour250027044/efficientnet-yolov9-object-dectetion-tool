import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        sa = torch.cat([torch.max(x, 1, keepdim=True)[0],
                        torch.mean(x, 1, keepdim=True)], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        out_channels = in_channels // 2

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1 if rate == 1 else 3,
                          padding=0 if rate == 1 else rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for rate in atrous_rates
        ])

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(atrous_rates), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        x = torch.cat(res, dim=1)
        return self.project(x)


class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2, eps=1e-4):
        super(BiFPN, self).__init__()
        self.eps = eps
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_layers)])
        self.ws = nn.Parameter(torch.ones(num_layers, 2))

    def forward(self, x):
        if isinstance(x, list) and len(x) == 3:
            P3, P4, P5 = x
            for i in range(self.num_layers):
                w = F.relu(self.ws[i])
                w /= (torch.sum(w, dim=0) + self.eps)
                P4 = w[0] * P4 + w[1] * F.interpolate(P5, size=P4.shape[2:], mode='nearest')
                P3 = w[0] * P3 + w[1] * F.interpolate(P4, size=P3.shape[2:], mode='nearest')
                P5 = w[0] * P5 + w[1] * F.max_pool2d(P4, kernel_size=2)
            return [P3, P4, P5]
        else:
            return x


class DropBlock(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, 0:1, :, :]) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        mask = mask.expand_as(x)
        return x * mask * (mask.numel() / mask.sum())
