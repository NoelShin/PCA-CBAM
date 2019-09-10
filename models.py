from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from attention_modules import BAM, CBAM, SqueezeExcitationBlock
from scbam import CCM, DBAM


class BasicConv(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, padding=0, stride=1, use_batchnorm=True, groups=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_ch, output_ch, kernel_size, padding, stride,
                                            bias=False if use_batchnorm else True, groups=groups),
                                     nn.BatchNorm2d(output_ch),
                                     nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self, width_multiplier=1.0, input_ch=3, init_ch=32, n_classes=1000):
        super(MobileNet, self).__init__()
        n_ch = int(init_ch * width_multiplier)
        self.network = nn.Sequential(BasicConv(input_ch, n_ch, 3, padding=1, stride=2),

                                     BasicConv(n_ch, n_ch, 3, padding=1, groups=n_ch),
                                     BasicConv(n_ch, 2 * n_ch, 1),

                                     BasicConv(2 * n_ch, 2 * n_ch, 3, padding=1, stride=2, groups=2 * n_ch),
                                     BasicConv(2 * n_ch, 4 * n_ch, 1),
                                     BasicConv(4 * n_ch, 4 * n_ch, 3, padding=1, groups=4 * n_ch),
                                     BasicConv(4 * n_ch, 4 * n_ch, 1),

                                     BasicConv(4 * n_ch, 4 * n_ch, 3, padding=1, stride=2, groups=4 * n_ch),
                                     BasicConv(4 * n_ch, 8 * n_ch, 1),
                                     BasicConv(8 * n_ch, 8 * n_ch, 3, padding=1, groups=8 * n_ch),
                                     BasicConv(8 * n_ch, 8 * n_ch, 1),

                                     BasicConv(8 * n_ch, 8 * n_ch, 3, padding=1, stride=2, groups=4 * n_ch),
                                     BasicConv(8 * n_ch, 16 * n_ch, 1),

                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 16 * n_ch, 1),
                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 16 * n_ch, 1),
                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 16 * n_ch, 1),
                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 16 * n_ch, 1),
                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 16 * n_ch, 1),

                                     BasicConv(16 * n_ch, 16 * n_ch, 3, padding=1, stride=2, groups=16 * n_ch),
                                     BasicConv(16 * n_ch, 32 * n_ch, 1),
                                     BasicConv(32 * n_ch, 32 * n_ch, 3, padding=1, groups=32 * n_ch),
                                     BasicConv(32 * n_ch, 32 * n_ch, 1),

                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     View(-1),
                                     nn.Linear(32 * n_ch, n_classes))

        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, bottle_neck_ch=0, pre_activation=False, first_conv_stride=1, n_groups=1,
                 attention='CBAM', conversion_factor=4, crop_boundary=(0, 0)):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d

        if pre_activation:
            if bottle_neck_ch:
                block = [norm(input_ch),
                         act,
                         nn.Conv2d(input_ch, bottle_neck_ch, 1, bias=False)]  # Caffe version has stride 2 here

                block += [norm(bottle_neck_ch),
                          act,
                          nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, padding=1, stride=first_conv_stride,
                                    groups=n_groups, bias=False)]  # PyTorch version has stride 2 here

                block += [norm(bottle_neck_ch),
                          nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False)]

            else:
                block = [norm(input_ch),
                         act,
                         nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, groups=n_groups, padding=1,
                                   bias=False)]

                block += [norm(output_ch),
                          act,
                          nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False)]

        else:
            if bottle_neck_ch:
                block = [nn.Conv2d(input_ch, bottle_neck_ch, 1, bias=False),  # Caffe version has stride 2 here
                         norm(bottle_neck_ch),
                         act]
                block += [nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, padding=1, stride=first_conv_stride,
                                    groups=n_groups,
                                    bias=False),  # PyTorch version has stride 2 here
                          norm(bottle_neck_ch),
                          act]
                block += [nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False),
                          norm(output_ch)]

            else:
                block = [nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, groups=n_groups, padding=1,
                                   bias=False),
                         norm(output_ch),
                         act]
                block += [nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False),
                          norm(output_ch)]

        if attention == 'CBAM':
            block += [CBAM(output_ch)]

        elif attention == 'SE':
            block += [SqueezeExcitationBlock(output_ch)]

        elif attention == 'CCM':
            block += [CCM(output_ch, conversion_factor=conversion_factor, crop_boundary=crop_boundary)]

        if input_ch != output_ch:
            side_block = [nn.Conv2d(input_ch, output_ch, 1, stride=first_conv_stride, bias=False),
                          norm(output_ch)]
            self.side_block = nn.Sequential(*side_block)
            self.varying_size = True

        else:
            self.varying_size = False

        self.block = nn.Sequential(*block)

    def forward(self, x):
        if self.varying_size:
            return F.relu(self.side_block(x) + self.block(x))
        else:
            return F.relu(x + self.block(x))


class ResidualNetwork(nn.Module):
    def __init__(self, n_layers=50, dataset='ImageNet', attention='CCM', conversion_factor=4):
        super(ResidualNetwork, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        RB = partial(ResidualBlock, attention=attention, conversion_factor=conversion_factor)

        if dataset == 'ImageNet':
            network = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                       norm(64),
                       act,
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            n_classes = 1000

        elif dataset =='CIFAR10':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       norm(64),
                       act]
            n_classes = 10

        elif dataset == 'CIFAR100':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       norm(64),
                       act]
            n_classes = 100

        else:
            """
            For other dataset
            """
            pass

        if n_layers == 18:
            network += [RB(64, 64),
                        RB(64, 64)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [RB(64, 128, first_conv_stride=2),
                        RB(128, 128)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [RB(128, 256, first_conv_stride=2),
                        RB(256, 256)]
            network += [BAM(256)] if attention == 'BAM' else []

            if dataset == 'ImageNet' and attention == 'CCM':
                network += [nn.ZeroPad2d((0, 1, 0, 1))]

            network += [RB(256, 512, first_conv_stride=2),
                        RB(512, 512)]
            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 34:
            network += [RB(64, 64)
                        for _ in range(3)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [RB(64, 128, first_conv_stride=2)]
            network += [RB(128, 128) for _ in range(3)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [RB(128, 256, first_conv_stride=2)]
            network += [RB(256, 256) for _ in range(5)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [RB(256, 512, first_conv_stride=2)]
            network += [RB(512, 512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 50:
            network += [RB(64, 256, bottle_neck_ch=64)]
            network += [RB(256, 256, bottle_neck_ch=64) for _ in range(2)]
            network += [BAM(256)] if attention == 'BAM' else []

            # if dataset == 'ImageNet' and attention == 'CCM':
            #     network += [nn.ZeroPad2d(4)]

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2, crop_boundary=(0, 0))]  # 28
            network += [RB(512, 512, bottle_neck_ch=128, crop_boundary=(0, 0)) for _ in range(3)]
            network += [BAM(512)] if attention == 'BAM' else []

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2, crop_boundary=(0, 0))]  # 14
            network += [RB(1024, 1024, bottle_neck_ch=256, crop_boundary=(0, 0)) for _ in range(5)]
            network += [BAM(1024)] if attention == 'BAM' else []

            # if dataset == 'ImageNet' and attention == 'CCM':
            #     network += [nn.ZeroPad2d((0, 1, 0, 1))]

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, crop_boundary=(0, 0))]
            network += [RB(2048, 2048, bottle_neck_ch=512, crop_boundary=(0, 0)) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 101:
            network += [RB(64, 256, bottle_neck_ch=64)]
            network += [RB(256, 256, bottle_neck_ch=64) for _ in range(2)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2)]
            network += [RB(512, 512, bottle_neck_ch=128) for _ in range(3)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2)]
            network += [RB(1024, 1024, bottle_neck_ch=256) for _ in range(22)]
            network += [BAM(256)] if attention == 'BAM' else []

            if dataset == 'ImageNet' and attention == 'CCM':
                network += [nn.ZeroPad2d((0, 1, 0, 1))]

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2)]
            network += [RB(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 152:
            network += [RB(64, 256, bottle_neck_ch=64)]
            network += [RB(256, 256, bottle_neck_ch=64) for _ in range(2)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2)]
            network += [RB(512, 512, bottle_neck_ch=128) for _ in range(7)]
            network += [BAM(512)] if attention == 'BAM' else []

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2)]
            network += [RB(1024, 1024, bottle_neck_ch=256) for _ in range(35)]
            network += [BAM(1024)] if attention == 'BAM' else []

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2)]
            network += [RB(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        else:
            raise NotImplementedError("Invalid n_layers {}. Choose among 18, 34, 50, 101, or 152.".format(n_layers))

        self.network = nn.Sequential(*network)
        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class ResNext(nn.Module):
    def __init__(self, n_layers=50, n_groups=32, dataset='ImageNet', attention='SE', conversion_factor=4):
        super(ResNext, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        RB = partial(ResidualBlock, attention=attention, n_groups=n_groups, conversion_factor=conversion_factor)
        if dataset == 'ImageNet':
            network = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                       norm(64),
                       act,
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            n_classes = 1000

        elif dataset == 'CIFAR100':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       norm(64),
                       act]
            n_classes = 100

        if n_layers == 50:
            network += [RB(64, 256, bottle_neck_ch=128)]
            network += [RB(256, 256, bottle_neck_ch=128) for _ in range(2)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [RB(256, 512, bottle_neck_ch=256, first_conv_stride=2)]  # 28
            network += [RB(512, 512, bottle_neck_ch=256) for _ in range(3)]
            network += [BAM(512)] if attention == 'BAM' else []

            network += [RB(512, 1024, bottle_neck_ch=512, first_conv_stride=2)]  # 14
            network += [RB(1024, 1024, bottle_neck_ch=512) for _ in range(5)]
            network += [BAM(1024)] if attention == 'BAM' else []

            if dataset == 'ImageNet' and attention == 'SeparableCBAM':
                network += [nn.ZeroPad2d((0, 1, 0, 1))]

            network += [RB(1024, 2048, bottle_neck_ch=1024, first_conv_stride=2)]
            network += [RB(2048, 2048, bottle_neck_ch=1024) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]
        self.network = nn.Sequential(*network)
        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class WideResNet(nn.Module):
    def __init__(self, n_layers=18, widening_factor=1.5, dataset='ImageNet', attention='None', conversion_factor=4):
        super(WideResNet, self).__init__()
        assert (n_layers - 4) % 6 == 0
        N = (n_layers - 4) // 6
        RB = partial(ResidualBlock, attention=attention, pre_activation=True, conversion_factor=conversion_factor)
        if dataset == 'ImageNet':
            n_classes = 1000
        elif dataset == 'CIFAR10':
            n_classes = 10
        elif dataset == 'CIFAR100':
            n_classes = 100

        network = [nn.Conv2d(3, 16, 3, padding=1, bias=False)]

        network += [RB(16, 16 * widening_factor)]
        for _ in range(N-1):
            network += [RB(16 * widening_factor, 16 * widening_factor)]

        network += [RB(16 * widening_factor, 32 * widening_factor)]
        for _ in range(N-1):
            network += [RB(32 * widening_factor, 32 * widening_factor)]

        network += [RB(32 * widening_factor, 64 * widening_factor)]
        for _ in range(N-1):
            network += [RB(64 * widening_factor, 64 * widening_factor)]

        network += [nn.BatchNorm2d(64 * widening_factor),
                    nn.ReLU(True)]

        network += [nn.AdaptiveAvgPool2d((1, 1)),
                    View(-1),
                    nn.Linear(64 * widening_factor, n_classes)]

        self.network = nn.Sequential(*network)
        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)


if __name__ == '__main__':
    #resnet = ResidualNetwork(50, dataset='ImageNet', attention='CCM', conversion_factor=4)
    #resnext = ResNext(50, dataset='ImageNet', attention='None')
    # wrn = WideResNet(28, dataset='CIFAR100', attention='None')
    mobilenet = MobileNet()

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(mobilenet, (3, 224, 224), as_strings=False, print_per_layer_stat=True)
    print('GFlops:  ', flops / (1.024 ** 3), '# Params: ', params)
