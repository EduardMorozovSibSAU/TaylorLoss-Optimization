'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Гиперпараметры
num_classes = 10


# CHANGE
root = "C:\\Users\\User\\Desktop"

# Custom loss Taylor
class TaylorLoss(nn.Module):
    def __init__(self, x):
        super(TaylorLoss, self).__init__()

    def forward(self, output, target, x):
         
        target = F.one_hot(target, num_classes)
        x = torch.from_numpy(x)
        sm = nn.Softmax(dim = 1)
        output = sm(output)
 
        # print(target)
        #print(f'{output}')
 
        summ = torch.sum( torch.add( target.sub(x[1]).mul(x[2]), target.sub(x[1]).pow(2).mul(x[3].mul(0.5)) ).add( target.sub(x[1]).pow(3).mul( x[4].mul(1/6) )).add( torch.mul(output.sub(x[0]),target.sub(x[1])).mul(x[5]) ).add( torch.mul( output.sub(x[0]), target.sub(x[1]).pow(2)).mul( x[6].mul(0.5) ) ).add( torch.mul( output.sub(x[0]).pow(2), target.sub(x[1]) ).mul( x[7].mul(0.5) )))
        summ = summ.mul( -1/num_classes )
        
        #print(f'\n После подсчетов \n')
        # print(target)
        #print(output)
        
        return summ




# Training
def train(epoch, model, trainloader, optimizer, criterion, x_massive):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        
        loss = criterion(outputs, targets, x_massive)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))







def test(epoch, model, testloader, criterion, x_massive, best_acc):
    # global best_acc
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    recall = 0
    precision = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets, x_massive)
          
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc


def TEST_model(x_massive, model, name):

    
    acc_temp = np.zeros(1)

    for i in range(1):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        args = parser.parse_args()

        
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=16)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=16)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        # Model

        print('==> Building model..')
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True


        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            model.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        # criterion = TaylorLoss()
        criterion = TaylorLoss(x_massive)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        for epoch in range(start_epoch, start_epoch+200):
            
            train(epoch, model, trainloader, optimizer, criterion, x_massive)
                
            acc_temp[i] = test(epoch, model, testloader, criterion, x_massive, best_acc) # Перезаписываем точность после каждого теста
                
            scheduler.step()

        np.savetxt( os.path.join(root,f"./models_TEST/{name}.txt"), acc_temp)



if __name__ == '__main__':
    
    #x_massive = np.array([2.595136825312013507e+00, -2.542575777074239429e+00, 2.101264109189781237e+00, -1.413129813887754160e+00, 3.539724155423960905e+00, 4.860361706605027976e+00, -4.561698585505042480e+00, -3.514057818820513379e+00]) # MOEA f1_acc LeNet

    x_massive = np.array([-7.847127499474330747e+00, -8.635872508062386599e-03, -7.277742100585238738e+00, 6.871386786766024102e+00, 7.534108368014847734e+00, 1.174391027020007083e+00, 4.625205946852370076e+00, -3.993614503566861873e-01]) # addit Train with Surr



    net = LeNet()
    TEST_model(x_massive, net, name = 'LeNet')


    ##net = VGG('VGG19')
    ##TEST_model(x_massive, net, name = 'VGG')


    net = ResNet18()
    TEST_model(x_massive, net, name = 'ResNet18')
    net = ResNet50()
    TEST_model(x_massive, net , name = 'ResNet50')
    net = ResNet101()
    TEST_model(x_massive, net, name = 'ResNet101')


    net = RegNetX_200MF()
    TEST_model(x_massive, net, name = 'RegNetX_200MF')
    net = RegNetY_400MF()
    TEST_model(x_massive, net, name = 'RegNetY_400MF')

    net = MobileNetV2()
    TEST_model(x_massive, net, name = 'MobileNetV2')


    net = ResNeXt29_32x4d()
    TEST_model(x_massive, net, name = 'ResNeXt29_32x4d')
    net = ResNeXt29_2x64d()
    TEST_model(x_massive, net, name = 'ResNeXt29_2x64d')


    net = SimpleDLA()
    TEST_model(x_massive, net, name = 'SimpleDLA')


    net = DenseNet121()
    TEST_model(x_massive, net, name = 'DenseNet121')


    net = PreActResNet18()
    TEST_model(x_massive, net, name = 'PreActResNet18')


    net = DPN92()
    TEST_model(x_massive, net, name = 'DPN92')


    net = DLA()
    TEST_model(x_massive, net, name = 'DLA')



    

    
