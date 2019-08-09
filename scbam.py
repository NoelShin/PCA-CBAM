import torch
import torch.nn.functional as F
import torch.nn as nn


class ChannelAxisPool(nn.Module):
    def __init__(self, pool='channelwise_conv', n_ch=None):
        super(ChannelAxisPool, self).__init__()
        assert pool in ['avg', 'channelwise_conv', 'max', 'var'], print("Invalid type {}. Choose among ['avg', 'max', 'var']".format(pool))
        if pool == 'channelwise_conv':
            assert n_ch, print("To use channelwise_conv, you need to clarify n_ch argument.")
            self.conv = nn.Conv2d(n_ch, 1, 1)
        self.pool = pool

    def forward(self, x):
        if self.pool == 'avg':
            return torch.mean(x, dim=1, keepdim=True)

        elif self.pool == 'channelwise_conv':
            return self.conv(x)

        elif self.pool == 'max':
            return torch.max(x, dim=1, keepdim=True)[0]

        else:
            return torch.var(x, dim=1, keepdim=True)


class GlobalVarPool2d(nn.Module):
    def __init__(self):
        super(GlobalVarPool2d, self).__init__()

    def forward(self, x):
        return torch.mean(x ** 2, dim=(2, 3), keepdim=True) - torch.mean(x, dim=(2, 3), keepdim=True) ** 2


class MixedSeparableConv2d(nn.Module):
    def __init__(self, n_ch, bias=False):
        super(MixedSeparableConv2d, self).__init__()
        self.n_ch = n_ch
        self.conv3 = nn.Conv2d(n_ch // 4, n_ch // 4, 3, padding=1, groups=n_ch // 4, bias=bias)
        self.conv5 = nn.Conv2d(n_ch // 4, n_ch // 4, 5, padding=2, groups=n_ch // 4, bias=bias)
        self.conv7 = nn.Conv2d(n_ch // 4, n_ch // 4, 7, padding=3, groups=n_ch // 4, bias=bias)
        self.conv9 = nn.Conv2d(n_ch // 4, n_ch // 4, 9, padding=4, groups=n_ch // 4, bias=bias)

    def forward(self, x):
        return torch.cat((self.conv3(x[:, :self.n_ch // 4, ...]),
                          self.conv5(x[:, self.n_ch // 4:self.n_ch // 2, ...]),
                          self.conv7(x[:, self.n_ch // 2:self.n_ch * 3 // 4, ...]),
                          self.conv9(x[:, self.n_ch * 3 // 4:, ...])), dim=1)


class SeparableCBAM(nn.Module):
    def __init__(self, n_ch, crop_boundary=None):
        super(SeparableCBAM, self).__init__()
        bias = True
        self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, stride=2, groups=n_ch, bias=bias),
                                               nn.ReLU(True),
                                               nn.Conv2d(n_ch, n_ch, 3, padding=1, stride=2, groups=n_ch, bias=bias),
                                               nn.ReLU(True),
                                               nn.Conv2d(n_ch, n_ch, 3, padding=1, stride=2, groups=n_ch, bias=bias),
                                               nn.ReLU(True),
                                               nn.PixelShuffle(2),
                                               nn.Conv2d(n_ch // 4, n_ch // 4, 9, padding=4, groups=n_ch // 4),
                                               nn.ReLU(True),
                                               nn.PixelShuffle(2),
                                               nn.Conv2d(n_ch // 16, n_ch // 16, 9, padding=4,  groups=n_ch // 16),
                                               nn.ReLU(True),
                                               nn.PixelShuffle(2),
                                               nn.Conv2d(n_ch // 64, n_ch // 64, 9, padding=4, groups=n_ch // 64),
                                               nn.ReLU(True),
                                               nn.Conv2d(n_ch // 64, 1, 1, bias=True))

        if crop_boundary:
            self.spatial_attention.add_module("Crop", Crop(crop_boundary))

    def forward(self, x):
        # discussion

        return x * torch.sigmoid(self.spatial_attention(x))


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Crop(nn.Module):
    def __init__(self, boundary=(0, 0)):
        super(Crop, self).__init__()
        self.crop = True if boundary != (0, 0) else False
        self.even = True if (boundary[0] % 2) == 0 else False
        self.y = boundary[0]
        self.x = boundary[1]

    def forward(self, x):
        if self.crop:
            if self.even:

                return x[:, :, self.y // 2:-(self.y // 2), self.x // 2:-(self.x // 2)]
            else:
                return x[:, :, :-self.y, :-self.x]
        else:
            return x
