import argparse


class BaseOptions(object):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name.')
    parser.add_argument('--residual_network_model', type=int, default=50, help="Choose among [18, 34, 50, 101]")

    @staticmethod
    def define_hyper_params(parser):
        if parser.dataset == 'ImageNet':
            parser.add_argument('--batch_size', type=int, default=256)
            parser.add_argument('--lr', type=float, default=0.1)
            parser.add_argument('--momentum', type=float, default=0.9)
            parser.add_argument('--weight_decay', type=float, default=1e-4)


    def parse(self):
        