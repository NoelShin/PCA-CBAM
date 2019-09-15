import torch.nn as nn


class TAM(nn.Module):
    def __init__(self, n_ch, conversion_factor):
        super(TAM, self).__init__()
        tam = []
        for i in range(conversion_factor):
            tam += [nn.Conv2d(n_ch, n_ch // 2, 1, groups=n_ch // 2)]
            n_ch //= 2
            if i != conversion_factor - 1:
                tam += [nn.PReLU(n_ch, init=0.0)]
        tam += [nn.Sigmoid()]
        self.tam = nn.Sequential(*tam)

    def forward(self, x):
        return x * self.tam(x)
