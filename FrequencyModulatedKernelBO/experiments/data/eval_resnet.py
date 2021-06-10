from typing import List, Dict

import os
import datetime
import argparse

import numpy as np

import torch
import torch.optim
import torch.backends.cudnn
import torchvision
import torchvision.transforms as transforms


from FrequencyModulatedKernelBO.experiments.config import data_dir_root, exp_dir_root
from FrequencyModulatedKernelBO.experiments.nas.resnet import ResNet, RESBLOCK_TYPE
from FrequencyModulatedKernelBO.experiments.data.exp_nas_resnet import N_CHANNEL_LIST, HP_CIFAR100, N_RESBLOCKS_IN_GROUP


NUM_WORKERS = 2


def load_cifar100(batch_size, pin_memory=False):
    data_type = 'CIFAR100'
    dataset = torchvision.datasets.CIFAR100

    transform_eval = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                                              std=np.array([63.0, 62.1, 66.7]) / 255.0)])
    transform_train = transforms.Compose([transforms.Pad(padding=4, padding_mode='reflect'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32),
                                          transform_eval])
    data_dir = os.path.join(data_dir_root(), data_type)

    train_data = dataset(root=data_dir, train=True, download=True, transform=transform_train)
    test_data = dataset(root=data_dir, train=False, download=True, transform=transform_eval)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                              num_workers=NUM_WORKERS, pin_memory=pin_memory)

    return train_loader, test_loader



def _evaluate_resnet_architecture(block_type_list_list: List[List[str]], dropout_p_list_list: List[List[float]],
                                  input_dim: List[int], n_channel_list: List[int], num_classes: int,
                                  hyper_params: Dict, filetag: str):
    use_gpu = True
    pin_memory = use_gpu
    data_type = hyper_params['data_type']
    batch_size = hyper_params['batch_size']
    max_epoch = hyper_params['max_epoch']
    lr = hyper_params['lr']
    weight_decay = hyper_params['weight_decay']
    dampening = hyper_params['dampening']
    momentum = hyper_params['momentum']
    milestones = hyper_params['milestones']
    gamma = hyper_params['gamma']

    model_str_list = []
    for block_type_list, dropout_p_list in zip(block_type_list_list, dropout_p_list_list):
        model_str_list.append(' --> '.join(['%3s (p = %.4f)' % (block_type, dropout_p)
                                            for block_type, dropout_p in zip(block_type_list, dropout_p_list)]))
    print('\n')
    print(' ' * 24 + 'WideResNet' + ' ' * 24)
    print('\n'.join(('>' * 22 + ' [Cell %d] %4d ' + '<' * 22 + '\n%s ') % (i + 1, channel_width, model_str)
                    for i, (channel_width, model_str) in enumerate(zip(N_CHANNEL_LIST, model_str_list))))
    print('\n')

    train_loader, test_loader = load_cifar100(batch_size=batch_size, pin_memory=pin_memory)
    model = ResNet(block_type_list_list=block_type_list_list, dropout_p_list_list=dropout_p_list_list,
                   n_input_channel=input_dim[0], n_channel_list=n_channel_list, num_classes=num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, dampening=dampening,
                                momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    print('WideResNet on %s data set (%d training), %d epochs with batch size of %d'
          % (data_type, len(train_loader.sampler), max_epoch, batch_size))

    model.initialize_params()
    if use_gpu:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
            print("***************************************\n"
                  "*********** %d GPUs are used ***********\n"
                  "***************************************" % torch.cuda.device_count())
        else:
            model = model.cuda()
    model_device = next(model.parameters()).device

    print('[%s]%3d epoch : Training begins' % (datetime.datetime.now().strftime('%H:%M:%S'), 0))
    for e in range(max_epoch):
        running_loss = 0
        for input_batch, output_batch in train_loader:
            if use_gpu:
                input_batch = input_batch.cuda(device=model_device)
                output_batch = output_batch.cuda(device=model_device)
            pred_batch = model(input_batch)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(pred_batch, output_batch, reduction='mean')
            loss.backward()
            optimizer.step()
            running_loss = 0.95 * running_loss + 0.05 * loss.item()
        if (e + 1) % 25 == 0:
            print('[%s]%3d epoch : Loss : %+.6E, lr : %.4f' % (datetime.datetime.now().strftime('%H:%M:%S'),
                                                               e + 1, running_loss, scheduler.get_last_lr()[0]))
        scheduler.step()

    time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_args = {'block_type_list_list': block_type_list_list,
                  'dropout_p_list_list': dropout_p_list_list,
                  'n_input_channel': input_dim[0],
                  'n_channel_list': n_channel_list,
                  'num_classes': num_classes}
    optimizer_args = {'lr': lr, 'weight_decay': weight_decay, 'dampening': dampening, 'momentum': momentum}
    scheduler_args = {'milestones': milestones, 'gamma': gamma}
    torch.save({'model_args': model_args,
                'optimizer_args': optimizer_args,
                'scheduler_args': scheduler_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
               f=os.path.join(exp_dir_root(), 'WideResNet-TESTEVALCHKP_%s_%s.pth'
                              % (filetag.replace('_', '-'), time_tag)))

    model.eval()
    n_data = 0
    n_correct = 0
    for input_batch, output_batch in test_loader:
        if use_gpu:
            input_batch = input_batch.cuda(device=model_device)
            output_batch = output_batch.cuda(device=model_device)
        pred_batch = model(input_batch)
        n_data += input_batch.size(0)
        n_correct += torch.sum(torch.argmax(pred_batch, dim=1) == output_batch).item()
    error = 1 - n_correct / n_data
    print(">>>>>>>>>>>>>>>>>>>>")
    print('Test Error %f' % error)
    print("<<<<<<<<<<<<<<<<<<<<")
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for architecture search [NASNet]')
    parser.add_argument('--architecture', dest='architecture', type=str, default=None,
                        help='[original, submission_best_single, supplementary_best_portfolio]')
    args = parser.parse_args()
    assert args.architecture in ['original', 'submission_best_single', 'supplementary_best_portfolio']

    architecture_dict = {
        'original': {
            'dropout_p':  [0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000],
            'block_type': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        },
        'submission_best_single': {
            'dropout_p':  [0.3008, 0.2992, 0.3114, 0.3024, 0.2962, 0.3026, 0.3031, 0.3009, 0.3024],
            'block_type': [0.0000, 0.0000, 0.0000, 0.0000, 2.0000, 0.0000, 0.0000, 2.0000, 0.0000]
        },
        'supplementary_best_portfolio': {
            'dropout_p':  [0.0916, 0.2672, 0.1919, 0.2463, 0.2649, 0.5148, 0.2833, 0.3584, 0.2516],
            'block_type': [0.0000, 3.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 3.0000]
        }
    }

    dropout_p = architecture_dict[args.architecture]['dropout_p']
    block_type = architecture_dict[args.architecture]['block_type']
    start_block = 0
    dropout_p_list_list = []
    block_type_list_list = []
    for n in N_RESBLOCKS_IN_GROUP:
        dropout_p_list_list.append(dropout_p[start_block:start_block + n])
        block_type_list_list.append([RESBLOCK_TYPE[int(elm)]
                                     for elm in block_type[start_block:start_block + n]])
        start_block += n
    test_error_list = []
    for _ in range(5):
        test_error = _evaluate_resnet_architecture(
            block_type_list_list=block_type_list_list, dropout_p_list_list=dropout_p_list_list,
            input_dim=HP_CIFAR100['input_dim'], n_channel_list=N_CHANNEL_LIST, num_classes=HP_CIFAR100['num_classes'],
            hyper_params=HP_CIFAR100, filetag=args.architecture)
        test_error_list.append(test_error)
    print('Mean   : %f' % np.mean(test_error_list))
    print('Median : %f' % np.median(test_error_list))
    print(test_error_list)
