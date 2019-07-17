import torch
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


class SigmoidMultiplication(nn.Module):
    def __init__(self):
        super(SigmoidMultiplication, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableCBAM(nn.Module):
    def __init__(self, n_ch):
        super(SeparableCBAM, self).__init__()
        # list_channel = []
        # list_spatial = []
        # for branch in branches:
        #     if branch == 'avg':
        #         list_channel += [nn.AdaptiveAvgPool2d(1)]
        #         list_spatial += [ChannelAxisPool('avg')]
        #
        #     elif branch == 'max':
        #         list_channel += [nn.AdaptiveMaxPool2d(1)]
        #         list_spatial += [ChannelAxisPool('max')]
        #
        #     elif branch == 'var':
        #         list_channel += [GlobalVarPool2d()]
        #         list_spatial += [ChannelAxisPool('var')]
        #
        #     else:
        #         raise NameError("Invalid branch name {}.".format(str(branch)))

        # self.separable_conv = nn.Sequential(nn.Conv2d(n_ch, n_ch, 7, padding=3, groups=n_ch),
        #                                     nn.Conv2d(n_ch, n_ch // 16, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Conv2d(n_ch // 16, n_ch, 1))

        # self.separable_conv = nn.Sequential(nn.Conv2d(n_ch, n_ch, 7, padding=3, groups=n_ch),
        #                                     nn.Conv2d(n_ch, n_ch//16, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Conv2d(n_ch//16, n_ch, 1))

        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // 16, 1),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch))

        # below is mix ch-space and space-mix
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, 1, 1),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(1, 1, 7, padding=3))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 7, padding=3, groups=n_ch),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch, 1, 1))

        # belwo is to use resnext architecture
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // 16, 1),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch // 16, 7, padding=3, groups=n_ch // 16),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch // 16, 7, padding=3, groups=n_ch // 16),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch, 1))

        # below is to use resnext archtecture for channel attention and use separable conv for spatial attention
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // 16, 1),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch // 16, 7, padding=3, groups=n_ch // 16),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch))

        # hmm (best model 5x5)
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // 16, 1),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.Conv2d(n_ch // 16, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 7, padding=3, groups=n_ch),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch, n_ch, 7, padding=3, groups=n_ch),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch, 1, 1))

        # light version hmm best model!!!! red 1r5r5r1
        reduction_ratio = 16
        kernel_size = 5
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True))
        #
        # self.channel_attention = nn.Conv2d(n_ch // reduction_ratio, n_ch, 1)
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch // 16, n_ch // 16, 3, padding=1, groups=n_ch // 16),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // 16, n_ch // 16, 3, padding=1, groups=n_ch // 16),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // 16, 1, 1))

        # version2. Accuracy rising speed is the fastest but end up normal accuracy(around 9)
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True),
        #                          nn.Conv2d(n_ch // 16, n_ch // 16, 5, padding=2, groups=n_ch // 16),
        #                          nn.ReLU(True),
        #                          nn.Conv2d(n_ch // 16, n_ch // 16, 5, padding=2, groups=n_ch // 16),
        #                          nn.ReLU(True))
        # self.channel_attention = nn.Conv2d(n_ch // reduction_ratio, n_ch, 1)
        # self.spatial_attention = nn.Conv2d(n_ch // reduction_ratio, 1, 1)

        # This is the best model!! kernel size 5 achieves 8.169... hhh
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch, 5, padding=2, groups=n_ch),
        #                          nn.ReLU(True),
        #                          nn.Conv2d(n_ch, n_ch, 5, padding=2, groups=n_ch),
        #                          nn.ReLU(True))
        #
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, n_ch, 1))
        # self.spatial_attention = nn.Conv2d(n_ch, 1, 1)

        # This is the best model for top 1 error. 26.780...
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                          nn.ReLU(True),
        #                          nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True))
        #
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio), nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio), nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, 1, 1))

        # Finally best.... 26.632 and 8.050 31red3r3r1 reduction ratio 16
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                          nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True))
        #
        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio), nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio), nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, 1, 1))

        # 31red3r3r1 only spatial model has the best... 25.86 and 7.822 >> let's analyze it
        # 31relu(and reduce by 16) 3relu3relu1(reduce to 1 channel)
        # reduction_ratio = 16
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                          nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True))
        #
        # #self.channel_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch, 1))
        #
        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=1,
        #                                                  groups=n_ch // reduction_ratio),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch // reduction_ratio, 1, 1))

        # Best model analyze 3r3r3
        reduction_ratio = 16
        # self.red = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
        #                          # nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
        #                          nn.ReLU(True))

        # self.channel_attention = nn.Sequential(nn.Conv2d(n_ch // reduction_ratio, n_ch, 1))

        # self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1,
        #                                                  groups=n_ch),
        #                                        nn.ReLU(True),
        #                                        nn.Conv2d(n_ch, n_ch, 3, padding=1,
        #                                                  groups=n_ch))
        #

        self.spatial_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
                                               nn.ReLU(True),
                                               nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch),
                                               nn.Conv2d(n_ch, 1, 1))

        self.channel_attention = nn.Sequential(nn.Conv2d(n_ch, n_ch // 16, 1),
                                               nn.ReLU(True),
                                               nn.Conv2d(n_ch // 16, n_ch, 1))

        # self.list_channel = list_channel
        # self.list_spatial = list_spatial

    def forward(self, x):
        # y = torch.zeros_like(x)

        # below is to use only tanh but this can magnify the magnitude as tanh has the range [-1, 1]
        # for i in range(len(self.list_channel)):
        #     y += self.separable_conv(self.list_channel[i](x) + self.list_spatial[i](x))
        #   return x * torch.tanh(y)

        # below is for both tanh and sigmoid. This can solve the magnitude problem by applying sigmoid after adding
        # tanh from all branches, but not cost effective.
        # for i in range(len(self.list_channel)):
        #     y += x * torch.tanh(self.list_channel[i](x) + self.list_spatial[i](x))
        # return x * torch.sigmoid(y)

        # below is not using sigmoid after tanh projection.
        # for i in range(len(self.list_channel)):
        #     y += self.separable_conv(self.list_channel[i](x) + self.list_spatial[i](x))
        # return x * torch.sigmoid(y)

        # below is using only separable conv and not using avg, max. and var axes.
        # return x * torch.sigmoid(self.separable_conv(x))

        # below is separately use channel and spatial attention.
        #return x * torch.sigmoid(self.channel_attention(x) + self.spatial_attention(x))

        # below is to apply sigmoid separately.
        #return x * torch.sigmoid(x * torch.sigmoid(self.channel_attention(x)) + x * torch.sigmoid(self.spatial_attention(x)))

        # below is mix ch-space and space-mix
        # return x * torch.sigmoid(self.channel_attention(x) + self.spatial_attention(x))

        # below is to use resnext architecture
        # return x * torch.sigmoid(self.channel_attention(x))

        # below is to use resnext archtecture for channel attention and use separable conv for spatial attention
        # return x * torch.sigmoid(self.channel_attention(x) + self.spatial_attention(x))

        # light version hmm
        #return x * torch.sigmoid(self.channel_attention(self.red(x)) + self.spatial_attention(self.red(x)))

        # discussion
        return x * torch.sigmoid(self.spatial_attention(x) + self.channel_attention(x))


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
