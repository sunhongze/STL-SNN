# train.py
#!/usr/bin/env	python3

""" GuoLab-UESTC
author: Hongze Sun
Institution: University of Electronic Science and Technology of China
Paper: A Synapse-Threshold Synergistic Learning Approach for Spiking Neural Networks
"""

import torch
import torch.nn.functional as F
import argparse
from spiking_model import *
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from params import mnist_para

def get_model_and_dataset(args):
    if args.data == 'mnist':
        ## load model
        model = MNIST()
        if args.gpu:
            model.cuda()

        ## load dataloader
        train_dataset = torchvision.datasets.MNIST(root= '', train=True, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=mnist_para.batch_size, shuffle=True, num_workers=mnist_para.batch_size)
        test_set = torchvision.datasets.MNIST(root= '', train=False, download=True,  transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=mnist_para.batch_size, shuffle=False, num_workers=mnist_para.num_worker)

        ## load loss function and optimizers
        criterion = F.mse_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=mnist_para.lr, weight_decay=1e-6, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=mnist_para.gama)

        ## get epoch
        epoch = mnist_para.epoch

        ## get batchsize
        batch_size = mnist_para.batch_size

    return model, train_loader, test_loader, criterion, optimizer, scheduler, epoch, batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default='mnist', help='mnist')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-tl', action='store_true', default=True, help='use the threshold learning')
    args = parser.parse_args()

    ## load the STL-SNN model, dataloader and other hyper-parameters.
    model, train_loader, test_loader, criterion, optimizer, scheduler, num_epoch, batch_size = get_model_and_dataset(args)

    ## train our STL-SNN model.
    for epoch in range(num_epoch):
        loss_train = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, F.one_hot(labels, 10).float())
            loss_train += loss.item()
            loss.backward()
            optimizer.step()

        print('Loss_train :',loss_train)
        scheduler.step()

        total = 10000
        correct = 0
        loss_test = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                print(batch_idx)
                if args.gpu:
                    inputs = inputs.cuda()
                outputs = model(inputs)
                labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                loss_test += loss.item()
                _, predicted = outputs.cpu().max(1)
                correct += float(predicted.eq(targets).sum().item())

        print('Loss_test  :', loss_test)
        print('Test Accuracy  : %.3f' % (100 * correct / total))

        print('Iters:', epoch + 1, '\n')
