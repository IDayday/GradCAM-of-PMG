from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import *


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root='./bird/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    checkpoint_path = './resnet50-19c8e357.pth'
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg',checkpoint_path=checkpoint_path, require_grad=True)
    netp = torch.nn.DataParallel(net, device_ids=[0,1])

    # GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    
    val_acc, val_acc_com, val_loss = test(net, CELoss, 3)
    print('test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (val_acc, val_acc_com, val_loss))



train(nb_epoch=40,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird',     # folder for output
         resume=True,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='./bird/model.pth')         # the saved model where you want to resume the training
