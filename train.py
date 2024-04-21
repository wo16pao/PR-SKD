import os
import random
import argparse
import torch.nn as nn
from time import time
from dataset import create_loader
import models
import distill_loss
from utils import *

def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./data/CIFAR100", type=str)
    parser.add_argument('--data', default='CIFAR100', type=str, help='CIFAR100|TINY|ImageNet|cub200|stanford40|dogs|mit67')
    parser.add_argument('--random_seed', default=10, type=int)

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--scheduler', default='step', type=str, help='step|cos')
    parser.add_argument('--schedule', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--model', default='cifarresnet18', type=str, help='cifarresnet18|wrn16x2|mobilenetv2|resnet18')

    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--width', default=2, type=int)
    parser.add_argument('--temperature', default=4, type=float)
    parser.add_argument('--alpha', default=3, type=float)
    parser.add_argument('--beta', default=100, type=float)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    args_path = '{}_{}_a={}_b={}]'.format(args.data, args.model, args.alpha, args.beta)
    path_log = os.path.join('logs', args_path)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = create_logging(os.path.join(path_log, '%s.txt' % args.random_seed))

    print(args)
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    train_loader, test_loader, args.num_classes = create_loader(args.batch_size, args.data_dir, args.data)

    model = models.__dict__[args.model](num_classes=args.num_classes)
    bifpns = []
    for i in range(len(model.network_channels) - 1):
        bifpns.append(models.BiFPN(model.network_channels[i:], args.num_classes, args))

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = distill_loss.att(args)
    criterion_kd.train()

    train_list = nn.ModuleList()
    train_list.append(model)
    train_list.append(criterion_ce)
    train_list.append(criterion_kd)
    for bifpn in bifpns:
        train_list.append(bifpn)
    train_list.cuda()

    criterion = [criterion_ce, criterion_kd]
    optimizer = optim.SGD(train_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler(optimizer, args.scheduler, args.schedule, args.lr_decay, args.epoch)

    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        t = time()
        loss, train_acc1 = train(model, bifpns, optimizer, criterion, train_loader)
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(path_log, "last.pt" % epoch))
        test_acc1 = test(model, test_loader)

        log_msg = 'Epoch: {0:>2d}|Train Loss: {1:2.4f}| Train Acc: {2:.4f}| Test Acc: {3:.4f}| Time: {4:4.2f}(s)'.format(epoch, loss, train_acc1, test_acc1, time() - t)
        
        if best_acc < test_acc1:
            best_acc = test_acc1
            torch.save(model.state_dict(), os.path.join(path_log, "best.pt"))
            log_msg += ' [*]'
            
        logger.info(log_msg)


if __name__ == '__main__':
    main()
