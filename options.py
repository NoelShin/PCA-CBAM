import os
import argparse


class BaseOptions(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--gpu_id', type=int, default=3)

        # Backbone options
        parser.add_argument('--backbone_network', type=str, default='ResNet',
                            help='Choose among [ResNet, WideResNet, ResNext]')
        parser.add_argument('--n_layers', type=int, default=50, help='# of weight layers.')

        # Attention options
        parser.add_argument('--attention_module', type=str, default='SCBAM',
                            help='Choose among [BAM, CBAM, None, SCBAM, SE]')
        parser.add_argument('--branches', type=str, default='avg, max', help='What branches you want to use for'
                                                                                  'SCBAM')
        parser.add_argument('--ordered', action='store_true', default=True, help='Channel-Spatial order')
        parser.add_argument('--scale', action='store_true', default=False)
        parser.add_argument('--shared_params', action='store_true', default=True, help='If you want to make SCBAM have'
                                                                                       'shared params between each'
                                                                                       'branch')

        parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name. Choose among'
                                                                            '[CIFAR100, ImageNet, MSCOCO, VOC2007]')
        parser.add_argument('--dir_checkpoints', type=str, default='./checkpoints')
        parser.add_argument('--dir_dataset', type=str, default='/DATA/RAID/Noel/Datasets/ImageNet-1K')
        parser.add_argument('--iter_report', type=int, default=5)
        parser.add_argument('--iter_save', type=int, default=100000)
        parser.add_argument('--n_workers', type=int, default=2)
        parser.add_argument('--residual_network_model', type=int, default=50, help="Choose among [18, 34, 50, 101]")

        self.parser = parser

    @staticmethod
    def define_hyper_params(args):
        if args.dataset == 'ImageNet':
            args.batch_size = 3  # default 256
            args.epochs = 90
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 1e-4

        elif args.dataset == 'CIFAR100':
            args.batch_size = 128
            args.epochs = 300
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 1e-4

    def parse(self):
        args = self.parser.parse_args()
        self.define_hyper_params(args)

        if args.dataset != 'ImageNet':
            args.dir_dataset = './datasets/{}'.format(args.dataset)

        model_name = args.backbone_network + str(args.n_layers) + '_' + args.attention_module
        if args.attention_module == 'SCBAM':
            model_name = model_name + '_' + str(args.branches).replace(', ', '_')
            model_name += '_shared_params_' if args.shared_params else ''
            model_name += '_scaled_' if args.scale else ''
            model_name += '_ordered_' if args.ordered else ''
        model_name = model_name.strip('_')

        args.dir_analysis = os.path.join(args.dir_checkpoints, args.dataset, model_name, 'Analysis')
        args.dir_model = os.path.join(args.dir_checkpoints, args.dataset, model_name, 'Model')
        os.makedirs(args.dir_analysis, exist_ok=True)
        os.makedirs(args.dir_model, exist_ok=True)

        args.path_log_analysis = os.path.join(args.dir_analysis, 'log.txt')
        with open(args.path_log_analysis, 'wt') as log:
            log.write('Epoch, Epoch_best_top1,  Epoch_best_top5, Training_loss, Top1_error, Top5_error\n')
            log.close()

        with open(os.path.join(args.dir_model, 'options.txt'), 'wt') as log:
            opt = vars(args)
            print('-' * 50 + 'Options' + '-' * 50)
            for k, v in sorted(opt.items()):
                print(k, v)
                log.write(str(k) + ', ' + str(v) + '\n')
            print('-' * 107)
            log.close()
        return args


if __name__ == '__main__':
    opt = BaseOptions().parse()
