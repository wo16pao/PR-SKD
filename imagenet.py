import os
import random
import argparse

import torch
import torch.nn as nn
from time import time
from dataset_v2 import create_loader
import models
import distill_loss
from utils_v2 import *
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./data/ImageNet", type=str)
    parser.add_argument('--data', default='ImageNet', type=str, help='CIFAR100|TINY|cub200|stanford40|dogs|mit67')
    parser.add_argument('--random_seed', default=3407, type=int)

    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--scheduler', default='step', type=str, help='step|cos')
    parser.add_argument('--schedule', default=[30, 60, 90], type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--model', default='resnet18', type=str, help='cifarresnet18|wrn16x2|mobilenetv2|resnet18')

    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--width', default=2, type=int)
    parser.add_argument('--temperature', default=4, type=float)
    parser.add_argument('--alpha', default=3, type=float)
    parser.add_argument('--beta', default=100, type=float)
    parser.add_argument('--pyramid', default=True, type=bool)
    parser.add_argument('--non_pyramid_num_bifpn', default=0, type=int)

    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')

    args = parser.parse_args()

    args_path = '{}_{}_a={}_b={}_pyramid={}_rank{}'.format(args.data, args.model, args.alpha, args.beta, args.pyramid, args.rank)
    path_log = os.path.join('imagenet_logs', args_path)
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = create_logging(os.path.join(path_log, '%s.txt' % args.random_seed))

    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    cudnn.deterministic = True

    dist.init_process_group("nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    train_loader, test_loader, args.num_classes, train_sampler = create_loader(args.batch_size, args.data_dir, args.data)
    
    model = models.__dict__[args.model](num_classes=args.num_classes)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    bifpns = []
    if args.pyramid:
        for i in range(len(model.module.network_channels) - 1):
            bifpn = models.BiFPN(model.module.network_channels[i:], args.num_classes, args)
            bifpn.cuda()
            bifpn = torch.nn.parallel.DistributedDataParallel(bifpn)
            bifpns.append(bifpn)
            # bifpns.append(models.BiFPN(model.network_channels[i:], args.num_classes, args))
    else:
        for i in range(args.non_pyramid_num_bifpn):
            bifpn = models.BiFPN(model.module.network_channels, args.num_classes, args)
            bifpn.cuda()
            bifpn = torch.nn.parallel.DistributedDataParallel(bifpn)
            bifpns.append(bifpn)
            # bifpns.append(models.BiFPN(model.network_channels, args.num_classes, args))
    # for bifpn in bifpns:
    #     bifpn.cuda()
    #     bifpn = torch.nn.parallel.DistributedDataParallel(bifpn)

    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion_kd = distill_loss.att(args)
    # criterion_kd.train()
    criterion_kd.cuda()

    train_list = nn.ModuleList()
    train_list.append(model)
    train_list.append(criterion_ce)
    train_list.append(criterion_kd)
    for bifpn in bifpns:
        train_list.append(bifpn)

    criterion = [criterion_ce, criterion_kd]
    optimizer = optim.SGD(train_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler(optimizer, args.scheduler, args.schedule, args.lr_decay, args.epoch)

    cudnn.benchmark = True

    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        t = time()
        train_sampler.set_epoch(epoch)
        loss, train_acc1, train_acc5 = train(model, bifpns, optimizer, criterion, train_loader)
        #loss, train_acc1, train_acc5 = train_v2(model, optimizer, criterion, train_loader)
        scheduler.step()
        # torch.save(model.state_dict(), os.path.join(path_log, "%s.pt" % epoch))
        test_acc1, test_acc5 = test(model, test_loader)
        
        log_msg = 'Epoch: {0:>2d}|Train Loss: {1:2.4f}| Train Acc1: {2:.4f}| Train Acc5: {3:.4f}| Test Acc: {4:.4f}| Test Acc5: {5:.4f}| Time: {6:4.2f}(s)'.format(epoch, loss, train_acc1, train_acc5, test_acc1, test_acc5, time() - t)
        
        if best_acc < test_acc1:
            best_acc = test_acc1
            torch.save(model.state_dict(), os.path.join(path_log, "best.pt"))
            log_msg += ' [*]'
            
        logger.info(log_msg)
        
        #logger.info('Epoch: {0:>2d}|Train Loss: {1:2.4f}| Train Acc1: {2:.4f}| Train Acc5: {3:.4f}| Test Acc: {4:.4f}| Test Acc5: {5:.4f}| Time: {6:4.2f}(s)'
        #            .format(epoch, loss, train_acc1, train_acc5, test_acc1, test_acc5, time() - t))


        
        
if __name__ == '__main__':
    main()