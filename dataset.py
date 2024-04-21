import os
import torch
from torchvision import transforms, datasets
from tiny_dataset import TinyImageNet


def create_loader(batch_size, data_dir, data):
    loader = {'CIFAR100': cifar_loader, 'TINY': tiny_loader, 'otherwise': other_loader}
    load_data = data if data in ['CIFAR100', 'TINY'] else 'otherwise'
    return loader[load_data](batch_size, data_dir, data)


def cifar_loader(batch_size, data_dir, data):
    num_label = 100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize, ])

    trainset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=True,
                                 download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=False,
                                download=True, transform=transform_test)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_label


def tiny_loader(batch_size, data_dir, data):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])

    trainset = TinyImageNet(data_dir, train=True, transform=transform_train)
    testset = TinyImageNet(data_dir, train=False, transform=transform_test)

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_label


def other_loader(batch_size, data_dir, data):
    print(data.lower())
    if data.lower() == 'cub200':
        num_label = 200
    elif data.lower() == 'dogs':
        num_label = 120
    elif data.lower() == 'mit67':
        num_label = 67
    elif data.lower() == 'stanford40':
        num_label = 40
    else:
        raise NotImplementedError('Dataset {} is not prepared.'.format(data))
    kwargs = {'num_workers': 0, 'pin_memory': True}
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), normalize, ])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                         transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_label
