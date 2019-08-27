def cal_top1_and_top5(output, label):
    batch_size = float(output.shape[0])
    _, index = output.topk(5, dim=1, largest=True, sorted=True)  # index shape: Bx5
    correct = index.eq(label.view(-1, 1).expand_as(index))  # correct shape: Bx5
    top1 = correct[:, :1].float().sum().mul_(100. / batch_size)
    top5 = correct[:, :5].float().sum().mul_(100. / batch_size)
    return 100.0 - top1, 100.0 - top5
