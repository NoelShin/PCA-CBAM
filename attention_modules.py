import torch
import torch.nn as nn


class AverageBranch(nn.Module):
    def __init__(self, n_ch):
        super(AverageBranch, self).__init__()
        self.channel_attention = SqueezeExcitationBlock(n_ch, pool='avg', sigmoid=False)
        self.spatial_attention = nn.Sequential(ChannelAxisPool(pool='avg'),
                                               nn.Conv2d(1, 1, 7, padding=3))

    def forward(self, x):
        return self.channel_attention(x) + self.spatial_attention(x)


class BAM(nn.Module):
    def __init__(self, n_ch, dilation=4, reduction_ratio=16):
        super(BAM, self).__init__()
        act = nn.ReLU(inplace=True)
        norm1d = nn.BatchNorm1d
        norm2d = nn.BatchNorm2d
        channel_branch = [nn.AdaptiveAvgPool2d(1),
                          View(-1),
                          nn.Linear(n_ch, n_ch // reduction_ratio),
                          act,
                          nn.Linear(n_ch // reduction_ratio, n_ch),
                          View(n_ch, 1, 1),
                          norm2d(n_ch)]
        spatial_branch = [nn.Conv2d(n_ch, n_ch // reduction_ratio, 1),
                          act]
        spatial_branch += [nn.Conv2d(n_ch // reduction_ratio, n_ch // reduction_ratio, 3, padding=4, dilation=dilation),
                           act] * 2
        spatial_branch += [nn.Conv2d(n_ch // reduction_ratio, 1, 1, bias=False), norm2d(1)]

        self.channel_branch = nn.Sequential(*channel_branch)
        self.spatial_branch = nn.Sequential(*spatial_branch)

    def forward(self, x):
        return x * (1 + torch.sigmoid(self.channel_branch(x) + self.spatial_branch(x)))


class CBAM(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.CBAM = nn.Sequential(ChannelAttentionModule(n_ch, reduction_ratio),
                                  SpatialAttentionModule())

    def forward(self, x):
        return self.CBAM(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        linear_A = nn.Linear(n_ch, n_ch // reduction_ratio)
        linear_B = nn.Linear(n_ch // reduction_ratio, n_ch)

        self.avg_descriptor = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            View(-1),
                                            linear_A,
                                            nn.ReLU(inplace=True),
                                            linear_B,
                                            View(n_ch, 1, 1))

        self.max_descriptor = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                            View(-1),
                                            linear_A,
                                            nn.ReLU(inplace=True),
                                            linear_B,
                                            View(n_ch, 1, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.avg_descriptor(x) + self.max_descriptor(x))


class ChannelAxisPool(nn.Module):
    def __init__(self, pool='avg'):
        super(ChannelAxisPool, self).__init__()
        assert pool in ['avg', 'max', 'var'], print("Invalid type {}. Choose among ['avg', 'max', 'var']".format(pool))
        self.pool = pool

    def forward(self, x):
        if self.pool == 'avg':
            return torch.mean(x, dim=1, keepdim=True)
        elif self.pool == 'max':
            return torch.max(x, dim=1, keepdim=True)[0]
        else:
            return torch.var(x, dim=1, keepdim=True)


class GlobalVarPool2d(nn.Module):
    def __init__(self, *shape):
        super(GlobalVarPool2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.mean(x ** 2, dim=(2, 3)) - torch.mean(x, dim=(2, 3)) ** 2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaxBranch(nn.Module):
    def __init__(self, n_ch):
        super(MaxBranch, self).__init__()
        self.channel_attention = SqueezeExcitationBlock(n_ch, pool='max', sigmoid=False)
        self.spatial_attention = nn.Sequential(ChannelAxisPool(pool='max'),
                                               nn.Conv2d(1, 1, 7, padding=3))

    def forward(self, x):
        return self.channel_attention(x) + self.spatial_attention(x)


class SCBAM(nn.Module):
    def __init__(self, n_ch, branches='var', scale=False, shared_params=False):
        super(SCBAM, self).__init__()
        assert isinstance(branches, (list, str, tuple))
        list_branches = nn.ModuleList()
        if isinstance(branches, str):
            assert branches in ['avg', 'max', 'var']
            if branches == 'avg':
                list_branches.append(AverageBranch(n_ch))
            elif branches == 'max':
                list_branches.append(MaxBranch(n_ch))
            else:
                list_branches.append(VarianceBranch(n_ch))
            self.list_branches = list_branches

        elif isinstance(branches, (list, tuple)):
            if shared_params:
                act = nn.ReLU(inplace=True)
                conv = nn.Conv2d(1, 1, 7, padding=3)
                linear_A = nn.Linear(n_ch, n_ch // 16)
                linear_B = nn.Linear(n_ch // 16, n_ch)

                channel_attentions = nn.ModuleList()
                spatial_attentions = nn.ModuleList()
                if 'avg' in branches:
                    channel_attentions.append(nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                            View(-1),
                                                            linear_A,
                                                            act,
                                                            linear_B,
                                                            View(n_ch, 1, 1)))

                    spatial_attentions.append(nn.Sequential(ChannelAxisPool('avg'), conv))

                if 'max' in branches:
                    channel_attentions.append(nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                                            View(-1),
                                                            linear_A,
                                                            act,
                                                            linear_B,
                                                            View(n_ch, 1, 1)))
                    spatial_attentions.append(nn.Sequential(ChannelAxisPool('max'), conv))

                if 'var' in branches:
                    channel_attentions.append(nn.Sequential(GlobalVarPool2d(),
                                                            View(-1),
                                                            linear_A,
                                                            act,
                                                            linear_B,
                                                            View(n_ch, 1, 1)))
                    spatial_attentions.append(nn.Sequential(ChannelAxisPool('var'), conv))

                self.channel_attentions = channel_attentions
                self.spatial_attentions = spatial_attentions

                self.norm = nn.BatchNorm2d(n_ch) if scale else None

            else:
                for branch in branches:
                    assert branch in ['avg', 'max', 'var']
                    if branch == 'avg':
                        if self.scale:
                            list_branches.append(nn.Sequential(AverageBranch(n_ch), nn.BatchNorm2d(n_ch)))
                        else:
                            list_branches.append(AverageBranch(n_ch))

                    elif branch == 'max':
                        if self.scale:
                            list_branches.append(nn.Sequential(MaxBranch(n_ch), nn.BatchNorm2d(n_ch)))
                        else:
                            list_branches.append(MaxBranch(n_ch))
                    else:
                        if self.scale:
                            list_branches.append(nn.Sequential(VarianceBranch(n_ch), nn.BatchNorm2d(n_ch)))
                        else:
                            list_branches.append(VarianceBranch(n_ch))
                self.list_branches = list_branches

        # self.ordered = ordered
        self.scale = scale
        self.shared_params = shared_params

    def forward(self, x):
        y = torch.zeros_like(x)
        if self.shared_params:
            for i in range(len(self.channel_attentions)):
                if self.scale:
                    y += self.norm(self.channel_attentions[i](x) + self.spatial_attentions[i](x))
                else:
                    y += self.channel_attentions[i](x) + self.spatial_attentions[i](x)

        else:
            for branch in self.list_branches:
                y += branch(x)

        return x * torch.sigmoid(y)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.avg_descriptor = ChannelAxisPool('avg')
        self.max_descriptor = ChannelAxisPool('max')
        self.block = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        return x * self.block(torch.cat((self.avg_descriptor(x), self.max_descriptor(x)), dim=1))


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, n_ch, reduction_ratio=16, pool='avg', sigmoid=True):
        super(SqueezeExcitationBlock, self).__init__()
        assert pool in ['avg', 'max', 'var']
        if pool == 'avg':
            block = [nn.AdaptiveAvgPool2d(1)]
        elif pool == 'max':
            block = [nn.AdaptiveMaxPool2d(1)]
        else:
            block = [GlobalVarPool2d(1)]

        block += [View(-1),
                  nn.Linear(n_ch, n_ch // reduction_ratio),
                  nn.ReLU(inplace=True),
                  nn.Linear(n_ch // reduction_ratio, n_ch), View(n_ch, 1, 1)]
        block += [nn.Sigmoid()] if sigmoid else []
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x * self.block(x)


class VarianceBranch(nn.Module):
    def __init__(self, n_ch):
        super(VarianceBranch, self).__init__()
        self.channel_attention = SqueezeExcitationBlock(n_ch, pool='var', sigmoid=False)
        self.spatial_attention = nn.Sequential(ChannelAxisPool(pool='var'),
                                               nn.Conv2d(1, 1, 7, padding=3))

    def forward(self, x):
        return self.channel_attention(x) + self.spatial_attention(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
