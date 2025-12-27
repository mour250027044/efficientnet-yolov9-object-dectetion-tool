import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_ch, out_ch, k=1, s=1, p=0, groups=1, act=True):
    """A convenience function for a convolutional layer followed by batch normalization and an optional activation."""
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False), nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_ch=None, se_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.se_ch = se_ch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None

    def forward(self, x):
        c = x.shape[1]
        if self.fc is None or c != self.in_ch:
            self.in_ch = c
            se_ch = max(1, c // 8) if self.se_ch is None else self.se_ch
            self.fc = nn.Sequential(
                nn.Conv2d(c, se_ch, 1, 1, 0), nn.SiLU(inplace=True), nn.Conv2d(se_ch, c, 1, 1, 0), nn.Sigmoid()
            ).to(x.device)
        return x * self.fc(self.pool(x))


class MyHGBlock(nn.Module):
    """Custom Hourglass-like block."""

    def __init__(self, in_ch=None, out_ch=None, expand_ratio=4, kernel=3, stride=1, use_se=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.use_se_flag = use_se
        self._built = False

    def build(self):
        """Builds the layers of the block dynamically based on input shape."""
        in_ch = self.in_ch
        out_ch = self.out_ch
        hidden_ch = int(in_ch * self.expand_ratio)
        pad = (self.kernel - 1) // 2

        self.expand_conv = (
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_ch), nn.SiLU(inplace=True)
            )
            if self.expand_ratio != 1
            else nn.Identity()
        )

        self.dw = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, self.kernel, self.stride, pad, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
        )

        if self.use_se_flag:
            self.se = SqueezeExcite(hidden_ch, max(1, in_ch // 8))
        else:
            self.se = nn.Identity()

        self.project = nn.Sequential(nn.Conv2d(hidden_ch, out_ch, 1, 1, 0, bias=False), nn.BatchNorm2d(out_ch))

        self.use_res_connect = self.stride == 1 and in_ch == out_ch
        self._built = True

    def forward(self, x):
        if not self._built:
            self.in_ch = x.shape[1]
            if self.out_ch is None:
                self.out_ch = self.in_ch
            self.build()

        out = self.expand_conv(x) if not isinstance(self.expand_conv, nn.Identity) else x
        out = self.dw(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out


class SPDADown(nn.Module):
    """Space-to-Depth-and-Attention Downsampling block."""

    def __init__(self, out_ch, block_size=2):
        super().__init__()
        self.block_size = block_size
        self.out_ch = out_ch
        self.pixel_unshuffle = nn.PixelUnshuffle(block_size)
        self.fuse = None

    def forward(self, x):
        h, w = x.shape[2:]
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.pixel_unshuffle(x)

        if self.fuse is None:
            in_ch = x.shape[1]
            self.fuse = conv_bn_act(in_ch, self.out_ch, k=1, s=1, p=0).to(x.device)
        return self.fuse(x)


class FABlock(nn.Module):
    """Feature Aggregation Block."""

    def __init__(self, in_ch=None, reduction=8):
        super().__init__()
        self.reduction = reduction
        self.se = SqueezeExcite(in_ch)
        self.s3 = None
        self.s5 = None
        self.s7 = None
        self.sig = nn.Sigmoid()
        self.fuse = None

    def forward(self, x):
        c = x.shape[1]

        if self.s3 is None:
            self.s3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False).to(x.device)
            self.s5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False).to(x.device)
            self.s7 = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False).to(x.device)
            self.reduce = nn.Conv2d(3 * c, c, 1, bias=False).to(x.device)
            self.fuse = conv_bn_act(c, c, k=1, s=1, p=0).to(x.device)

        x_ch = self.se(x)

        m3, m5, m7 = self.s3(x_ch), self.s5(x_ch), self.s7(x_ch)
        m = torch.cat([m3, m5, m7], dim=1)
        m = self.reduce(m)
        m = self.sig(m)

        out = x_ch * m + x_ch
        return self.fuse(out)


class Adapter(nn.Module):
    """Adapter block for channel transformation."""

    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch  # This was the original typo location, now corrected.
        self.proj = None

    def forward(self, x):
        c = x.shape[1]
        if self.proj is None or c != self.in_ch:
            self.in_ch = c
            if self.out_ch is None:
                self.out_ch = c
            self.proj = conv_bn_act(c, self.out_ch, k=1, s=1, p=0).to(x.device)
        return self.proj(x)
