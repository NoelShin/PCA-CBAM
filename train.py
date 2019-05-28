if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    import numpy as np
    from options import BaseOptions
    from pipeline import CustomImageNet1K
    from utils import cal_top1_and_top5
    from tqdm import tqdm
    from datetime import datetime

    opt = BaseOptions().parse()
    dataset_name = opt.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    torch.backends.cudnn.benchmark = True

    if dataset_name == 'CIFAR100':
        from pipeline import CustomCIFAR100
        dataset = CustomCIFAR100(opt, val=False)
        test_dataset = CustomCIFAR100(opt, val=True)

    elif dataset_name == 'ImageNet':
        from pipeline import CustomImageNet1K
        dataset = CustomImageNet1K(opt, val=False)
        test_dataset = CustomImageNet1K(opt, val=True)
    else:
        raise NotImplementedError("Invalid dataset {}. Choose among ['CIFAR100', 'ImageNet']".format(dataset_name))

    data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.n_workers, shuffle=False)

    backbone_network = opt.backbone_network
    n_layers = opt.n_layers

    if backbone_network == 'ResNet':
        from models import ResidualNetwork, init_weights
        model = ResidualNetwork(n_layers=n_layers,
                                dataset=opt.dataset,
                                attention=opt.attention_module,
                                branches=list(opt.branches.split(', ')),
                                shared_params=opt.shared_params).apply(init_weights).to(device)
    else:
        """
        Other models
        """

    criterion = nn.CrossEntropyLoss()
    if dataset_name == 'CIFAR100':
        optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                nesterov=True)
        lr_scheduler = MultiStepLR(optim, milestones=[150, 225], gamma=0.1)
    elif dataset_name == 'ImageNet':
        optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        lr_scheduler = MultiStepLR(optim, milestones=[30, 60], gamma=0.1)
    else:
        """
                For other datasets
        """
        raise NotImplementedError

    dict_best_top1 = {'Epoch': 0, 'Top1': 100.}
    dict_best_top5 = {'Epoch': 0, 'Top5': 100.}

    st = datetime.now()
    iter_total = 0
    for epoch in range(opt.epochs):
        list_loss = list()
        for input, label in tqdm(data_loader):
            iter_total += 1
            input, label = input.to(device), label.to(device)

            output = model(input)

            loss = criterion(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            top1, top5 = cal_top1_and_top5(output, label)

            list_loss.append(loss.detach().item())

            if iter_total % opt.iter_report == 0:
                print("Iteration: {}, Top1: {:.3f}, Top5: {:.3f} Loss: {:.4f}"
                      .format(iter_total, top1, top5, loss.detach().item()))

            if opt.debug:
                break

        with torch.no_grad():
            list_top1, list_top5 = list(), list()

            for input, label in tqdm(test_data_loader):
                input, label = input.to(device), label.to(device)

                output = model(input)

                top1, top5 = cal_top1_and_top5(output, label)
                list_top1.append(top1.cpu().numpy())
                list_top5.append(top5.cpu().numpy())

            avg_top1, avg_top5 = np.mean(list_top1), np.mean(list_top5)

            if avg_top1 < dict_best_top1['Top1']:
                dict_best_top1.update({'Epoch': epoch + 1, 'Top1': avg_top1})
                state = {'epoch': epoch + 1,
                         'lr': lr_scheduler.state_dict(),
                         'state_dict': model.state_dict(),
                         'optimizer': optim.state_dict()}
                torch.save(state, os.path.join(opt.dir_model, 'top1_best.pt'.format(epoch + 1)))

            if avg_top5 < dict_best_top5['Top5']:
                dict_best_top5.update({'Epoch': epoch + 1, 'Top5': avg_top5})
                state = {'epoch': epoch + 1,
                         'lr': lr_scheduler.state_dict(),
                         'state_dict': model.state_dict(),
                         'optimizer': optim.state_dict()}
                torch.save(state, os.path.join(opt.dir_model, 'top5_best.pt'.format(epoch + 1)))

            with open(os.path.join(opt.dir_analysis, 'log.txt'), 'a') as log:
                log.write(str(epoch + 1) + ', ' +
                          str(dict_best_top1['Epoch']) + ', ' +
                          str(dict_best_top5['Epoch']) + ', ' +
                          str(np.mean(list_loss)) + ', ' +
                          str(np.mean(list_top1)) + ', ' +
                          str(np.mean(list_top5)) + '\n')
                log.close()

        if opt.debug:
            break

        lr_scheduler.step()

    print("Total time taken: ", datetime.now() - st)
