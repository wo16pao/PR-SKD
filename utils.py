import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


def create_logging(path_log):
    logger = logging.getLogger('Result_log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


def train(model, bifpns, optimizer, criterion, train_loader):
    model.train()
    for bifpn in bifpns:
        bifpn.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    num_features = len(model.network_channels)

    criterion_ce, criterion_kd = criterion
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        feats, outputs = model(inputs)
        loss = criterion_ce(outputs, targets)

        for i, bifpn in enumerate(bifpns):
            b_feats, b_outputs = bifpn(feats[-num_features+i:])
            b_loss = criterion_ce(b_outputs, targets)
            f_loss = criterion_kd(outputs, b_outputs, feats[-num_features+i:], b_feats)
            loss += b_loss + f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1, batch_size)

    return losses.avg, top1.avg


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = targets.size(0)
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                feats, outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1, batch_size)
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def lr_scheduler(optimizer, scheduler, schedule, lr_decay, total_epoch):
    optimizer.zero_grad()
    optimizer.step()
    if scheduler == 'step':
        return optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=lr_decay)
    elif scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    else:
        raise NotImplementedError('{} learning rate is not implemented.')
        
        
def ablation_train(teacher, student, optimizer, criterion, train_loader, position):
    student.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    criterion_ce, criterion_kd = criterion

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            t_feats, _ = teacher(inputs)
        s_feats, s_outputs = student(inputs)
        t_feats = t_feats[position]
        s_feats = s_feats[position]
        loss = criterion_ce(s_outputs, targets)
        loss += criterion_kd(s_feats, t_feats)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        acc1, acc5 = accuracy(s_outputs, targets, topk=(1, 5))
        top1.update(acc1, batch_size)

    return losses.avg, top1.avg

