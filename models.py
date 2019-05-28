import torch.nn as nn
import torch.nn.functional as F
from attention_modules import BAM, CBAM, SCBAM, SqueezeExcitationBlock


class ResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, bottle_neck_ch=0, first_conv_stride=1, attention='CBAM', branches=None,
                 shared_params=True):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d

        if bottle_neck_ch:
            block = [nn.Conv2d(input_ch, bottle_neck_ch, 1, stride=first_conv_stride, bias=False),
                     norm(bottle_neck_ch),
                     act]
            block += [nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, padding=1, bias=False),
                      norm(bottle_neck_ch),
                      act]
            block += [nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False),
                      norm(output_ch)]

        else:
            block = [nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, padding=1, bias=False),
                     norm(output_ch),
                     act]
            block += [nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False),
                      norm(output_ch)]

        if attention == 'CBAM':
            block += [CBAM(output_ch)]

        elif attention == 'SCBAM':
            block += [SCBAM(output_ch, branches, shared_params)]

        elif attention == 'SE':
            block += [SqueezeExcitationBlock(output_ch)]

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
    def __init__(self, n_layers=50, dataset='ImageNet', attention='CBAM', branches=None, shared_params=True):
        super(ResidualNetwork, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        if dataset == 'ImageNet':
            network = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                       norm(64),
                       act,
                       nn.MaxPool2d(kernel_size=3, stride=2)]
            n_classes = 1000

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
            network += [ResidualBlock(64, 64, attention=attention, branches=branches, shared_params=shared_params),
                        ResidualBlock(64, 64, attention=attention, branches=branches, shared_params=shared_params)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [ResidualBlock(64, 128, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params),
                        ResidualBlock(128, 128, attention=attention, branches=branches, shared_params=shared_params)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [ResidualBlock(128, 256, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params),
                        ResidualBlock(256, 256, attention=attention, branches=branches, shared_params=shared_params)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [ResidualBlock(256, 512, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params),
                        ResidualBlock(512, 512, attention=attention, branches=branches, shared_params=shared_params)]
            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 34:
            network += [ResidualBlock(64, 64, attention=attention, branches=branches, shared_params=shared_params)
                        for _ in range(3)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [ResidualBlock(64, 128, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(128, 128, attention=attention, branches=branches, shared_params=shared_params)
                        for _ in range(3)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [ResidualBlock(128, 256, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(256, 256, attention=attention, branches=branches, shared_params=shared_params)
                        for _ in range(5)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [ResidualBlock(256, 512, first_conv_stride=2, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(512, 512, attention=attention, branches=branches, shared_params=shared_params)
                        for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 50:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(2)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(3)]
            network += [BAM(512)] if attention == 'BAM' else []

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256, attention=attention,
                                      branches=branches, shared_params=shared_params) for _ in range(5)]
            network += [BAM(1024)] if attention == 'BAM' else []

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 101:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(2)]
            network += [BAM(64)] if attention == 'BAM' else []

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(3)]
            network += [BAM(128)] if attention == 'BAM' else []

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256, attention=attention,
                                      branches=branches, shared_params=shared_params) for _ in range(22)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512, attention=attention,
                                      branches=branches, shared_params=shared_params) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 152:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(2)]
            network += [BAM(256)] if attention == 'BAM' else []

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(7)]
            network += [BAM(512)] if attention == 'BAM' else []

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(35)]
            network += [BAM(1024)] if attention == 'BAM' else []

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, attention=attention,
                                      branches=branches, shared_params=shared_params)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512, attention=attention, branches=branches,
                                      shared_params=shared_params) for _ in range(2)]

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
        nn.init.zeros_(module.bias)


if __name__ == '__main__':
    resnet = ResidualNetwork(50, dataset='CIFAR100', attention='None', shared_params=True, branches=['avg', 'max', 'var'])
    from ptflops import get_model_complexity_info
    from time import time

    st = time()
    flops, params = get_model_complexity_info(resnet, (3, 32, 32), as_strings=False, print_per_layer_stat=True)
    print(time() - st, "flops: ", flops, "params: ", params)
