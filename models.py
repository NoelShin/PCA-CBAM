import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, bottle_neck_ch=0, first_conv_stride=1):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        pad = nn.ZeroPad2d

        if bottle_neck_ch:
            block = [nn.Conv2d(input_ch, bottle_neck_ch, 1, stride=first_conv_stride, bias=False), norm(bottle_neck_ch),
                     act]
            block += [pad(1), nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, bias=False), norm(bottle_neck_ch), act]
            block += [nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False), norm(output_ch)]

        else:
            block = [pad(1), nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, bias=False), norm(output_ch),
                     act]
            block += [pad(1), nn.Conv2d(output_ch, output_ch, 3, bias=False), norm(output_ch)]

        if input_ch != output_ch:
            side_block = [nn.Conv2d(input_ch, output_ch, 1, stride=first_conv_stride, bias=False), norm(output_ch)]
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
    def __init__(self, n_layers=50):
        super(ResidualNetwork, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d
        pad = nn.ZeroPad2d
        network = [pad(3), nn.Conv2d(3, 64, 7, stride=2), norm(64), act, nn.MaxPool2d(kernel_size=3, stride=2)]
        if n_layers == 18:
            network += [ResidualBlock(64, 64), ResidualBlock(64, 64)]
            network += [ResidualBlock(64, 128, first_conv_stride=2), ResidualBlock(128, 128)]
            network += [ResidualBlock(128, 256, first_conv_stride=2), ResidualBlock(256, 256)]
            network += [ResidualBlock(256, 512, first_conv_stride=2), ResidualBlock(512, 512)]
            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, 1000)]

        elif n_layers == 34:
            network += [ResidualBlock(64, 64) for _ in range(3)]

            network += [ResidualBlock(64, 128, first_conv_stride=2)]
            network += [ResidualBlock(128, 128) for _ in range(3)]

            network += [ResidualBlock(128, 256, first_conv_stride=2)]
            network += [ResidualBlock(256, 256) for _ in range(5)]

            network += [ResidualBlock(256, 512, first_conv_stride=2)]
            network += [ResidualBlock(512, 512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, 1000)]

        elif n_layers == 50:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64) for _ in range(2)]

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128) for _ in range(3)]

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256) for _ in range(5)]

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, 1000)]

        elif n_layers == 101:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64) for _ in range(2)]

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128) for _ in range(3)]

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256) for _ in range(22)]

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, 1000)]

        elif n_layers == 152:
            network += [ResidualBlock(64, 256, bottle_neck_ch=64)]
            network += [ResidualBlock(256, 256, bottle_neck_ch=64) for _ in range(2)]

            network += [ResidualBlock(256, 512, bottle_neck_ch=128, first_conv_stride=2)]
            network += [ResidualBlock(512, 512, bottle_neck_ch=128) for _ in range(7)]

            network += [ResidualBlock(512, 1024, bottle_neck_ch=256, first_conv_stride=2)]
            network += [ResidualBlock(1024, 1024, bottle_neck_ch=256) for _ in range(35)]

            network += [ResidualBlock(1024, 2048, bottle_neck_ch=512, first_conv_stride=2)]
            network += [ResidualBlock(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, 1000)]

        else:
            print("Invalid n_layers {}. Choose among 18, 34, 50, 101, or 152.".format(n_layers))

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
        nn.init.kaiming_normal_(module.weight.detach(), mode='fan_out', nonlinearity='relu')

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight.detach())
        nn.init.zeros_(module.bias.detach())


if __name__ == '__main__':
    resnet = ResidualNetwork(152)
    from thop import profile
    from time import time

    st = time()
    flops, params = profile(resnet, input_size=(1, 3, 224, 224))
    print(time() - st, int(flops), int(params))