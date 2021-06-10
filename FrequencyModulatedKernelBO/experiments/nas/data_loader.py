import os
import numpy as np

import torch
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from ..config import data_dir_root


N_VALID_CIFAR = 10000
N_VALID_FashionMNIST = 30000
NUM_WORKERS = 2


def load_data(data_type: str, batch_size, shuffle=True, random_seed=0, pin_memory=False):
    if data_type in ['CIFAR10', 'CIFAR100']:
        transform_eval = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                                                  std=np.array([63.0, 62.1, 66.7]) / 255.0)])
        transform_train = transforms.Compose([transforms.Pad(padding=4, padding_mode='reflect'),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32),
                                              transform_eval])
    elif data_type in ['FashionMNIST']:
        transform_eval = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transform_eval])
    else:
        raise NotImplementedError
    data_dir = os.path.join(data_dir_root(), data_type)
    dataset = getattr(torchvision.datasets, data_type)

    train_data = dataset(root=data_dir, train=True, download=True, transform=transform_train)
    valid_data = dataset(root=data_dir, train=True, download=True, transform=transform_eval)
    test_data = dataset(root=data_dir, train=False, download=True, transform=transform_eval)
    indices = list(range(len(train_data)))
    if shuffle:
        np.random.RandomState(random_seed).shuffle(indices)
    if data_type in ['CIFAR10', 'CIFAR100']:
        train_idx, valid_idx = indices[:-N_VALID_CIFAR], indices[-N_VALID_CIFAR:]
    elif data_type == 'FashionMNIST':
        train_idx, valid_idx = indices[:-N_VALID_FashionMNIST], indices[-N_VALID_FashionMNIST:]
    else:
        raise NotImplementedError
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=NUM_WORKERS, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
