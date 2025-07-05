'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import math
import random
import numpy as np
import getpass
from numpy import float32

from models import *
from utils import progress_bar

from sklearn.metrics import precision_score as Precision
from sklearn.metrics import recall_score as Recall

from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.utils.design_space import DesignSpace
import matplotlib.pyplot as plt





device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHANGE
root = "C:\\Users\\User\\Desktop"


num_classes = 10

def euqlid_dist(popul): # a function for calculating the Euclidean distance.
    summ = 0
    count = 0
    for i in range(popul.shape[0]):
        for j in range(i+1):
            summ += math.dist(popul[i], popul[j])
            count += 1
    return summ/count


def Aquisition_LSRDE(reg,LeftPRS_,RightPRS_): 
    NInds = NIndsMax = 20 #NVars*3 number of Individuals
    NGens_sur = 20 # Number of Generations
    Fitness,FitTemp = np.zeros(NInds), np.zeros(NInds)
    PopulTemp = np.zeros((NInds,NVars))
    Popul = np.random.uniform(LeftPRS_,RightPRS_,size=(NInds,NVars))
    Arch = np.random.uniform(LeftPRS_,RightPRS_,size=(NInds,NVars))
    PopArch = np.zeros((NInds,NVars))
    f_min = np.min(yt)

    Pval = 0.3
    MeanF = 0.5
    MeanCr = 0.9
    SRTC = 0.25
    ArchProb = 0.5
    SelPress = 3

    for i in range(NInds):
        val = reg.predict_values(Popul[i:i+1,:])
        stdd = reg.predict_variances(Popul[i:i+1,:])


        # The value of the objective function is calculated
        Fitness[i] = val - stdd*3.0


    globalbest = np.min(Fitness)
    for gen in range(NGens_sur):
        indexes = np.argsort(Fitness)
        cur_i = np.arange(0,NInds,1)
        F_DE = np.random.standard_cauchy(NInds)*0.1+MeanF
        while(any(F_DE < 0.0)):
            F_DE[np.where(F_DE < 0)] = np.random.standard_cauchy()*0.1+MeanF
        F_DE = np.clip(F_DE,0,1)
        F_DE = np.reshape(np.repeat(F_DE,NVars),(NInds,NVars))
        Cr = np.clip(np.random.normal(0,0.1,NInds) + MeanCr,0,1)
        Cr = np.reshape(np.repeat(Cr,NVars),(NInds,NVars))
        jrand = np.random.randint(0,NVars,NInds)
        CrChange = Cr < np.random.uniform(0,1,(NInds,NVars))
        CrChange[cur_i,jrand] = False
        pbest = indexes[np.random.randint(0,Pval*NInds,NInds)]
        prob = np.exp(-cur_i/NInds*SelPress)
        prob /= np.sum(prob)
        r1_SP = np.random.choice(indexes, NInds, p=prob)
        WhereArch = ArchProb < np.random.uniform(0,1,(NInds))
        NArch = np.sum(WhereArch)
        r2 = np.random.randint(0,NInds,NInds)
        PopArch[WhereArch] = Arch[WhereArch]
        PopArch[np.logical_not(WhereArch)] = Popul[np.logical_not(WhereArch)]
        PopulTemp[cur_i] = Popul[cur_i] + F_DE*(Popul[pbest] - Popul[cur_i] + Popul[r1_SP] - PopArch[r2])
        PopulTemp[CrChange] = Popul[CrChange]

        for j in range(NVars):
            PopulTemp[np.where(PopulTemp[:,j] < LeftPRS[j]),j] = (Popul[np.where(PopulTemp[:,j] < LeftPRS[j]),j] + LeftPRS[j])*0.5
            PopulTemp[np.where(PopulTemp[:,j] > RightPRS[j]),j] = (Popul[np.where(PopulTemp[:,j] > RightPRS[j]),j] + RightPRS[j])*0.5

        for i in range(NInds):
        
            val = reg.predict_values(PopulTemp[i:i+1,:])
            stdd = reg.predict_variances(PopulTemp[i:i+1,:])

            FitTemp[i] = val - stdd*3.0
       
        globalbest = np.min([globalbest,np.min(FitTemp)])
        SuccessRate = np.sum(FitTemp < Fitness) / NInds
        repl = np.zeros(NInds,dtype = 'bool')
        for i in range(NInds):
            repl[i] = FitTemp[i] < Fitness[i]
        rand_i = np.random.randint(0,NInds,np.sum(repl))
        #Is_sur[repl] = Is_sur_temp[repl]
        Arch[rand_i] = Popul[repl]
        Popul[repl] = PopulTemp[repl]
        Fitness[repl] = FitTemp[repl]
        MeanF = np.power(SuccessRate,SRTC)
      
    besti = np.argmin(Fitness)
    fit_new, accuracy = target_func(Popul[besti:besti+1])
    return Popul[besti], fit_new, accuracy




# Custom loss Taylor
class TaylorLoss(nn.Module):
    def __init__(self, x):
        super(TaylorLoss, self).__init__()

    def forward(self, output, target, x):
         
        target = F.one_hot(target, num_classes)
        x = torch.from_numpy(x)
        
        sm = nn.Softmax(dim = 1)
        output = sm(output)
 
 
        summ = torch.sum( torch.add( target.sub(x[1]).mul(x[2]), target.sub(x[1]).pow(2).mul(x[3].mul(0.5)) ).add( target.sub(x[1]).pow(3).mul( x[4].mul(1/6) )).add( torch.mul(output.sub(x[0]),target.sub(x[1])).mul(x[5]) ).add( torch.mul( output.sub(x[0]), target.sub(x[1]).pow(2)).mul( x[6].mul(0.5) ) ).add( torch.mul( output.sub(x[0]).pow(2), target.sub(x[1]) ).mul( x[7].mul(0.5) )))
        summ = summ.mul( -1/num_classes )
    
        return summ

# Training
def train(epoch, net, trainloader, optimizer, criterion, x_massive):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets, x_massive)
        
        loss.backward()
        optimizer.step()
        
    
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
        

# Test model
def test(epoch, net, testloader, criterion, best_acc):
    # global best_acc
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    recall = 0
    precision = 0
    counter = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            counter += 1
            
            
            loss = criterion(outputs, targets)
            


            test_loss += loss.item()
            val_loss = test_loss/(batch_idx+1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print(f'Valid Loss = {val_loss}')

    # Save checkpoint.
    acc = 100.*correct/total
    
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        
        
    return val_loss, acc
    
   
   
def func_NN_calc(x):
    print(f'x shape = {x.shape}')
    f_ab = np.zeros((x.shape[0]), dtype=float32)
    
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
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    for lr in range(x.shape[0]):
        print(f'Обучение с {lr + 1} набором коэфициентов...')
        print(f'{x[lr]}')

        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        args = parser.parse_args()

        
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

       

        # Model
        print('==> Building model..')
        net = ResNet18() # from models import *
      
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion_Taylor = TaylorLoss(x[lr]) # Loss Function
        criterion_Cross_Entropy = nn.CrossEntropyLoss() #
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
        for epoch in range(start_epoch, start_epoch+200):
            # Процесс обучения
            train(epoch, net, trainloader, optimizer, criterion_Taylor, x[lr])
            
            f_ab[lr], flag_stop = test(epoch, net, testloader, criterion_Cross_Entropy, best_acc) # We rewrite the accuracy after each test
            
            scheduler.step()
            
            # Early stoppig if accuracy low            
            if epoch >= 2 and flag_stop <= 11:
                break


    return f_ab, flag_stop
    





if __name__ == '__main__':
    
    start_timer = time.time()
    
    n_sur = 1 # Number of points for the surrogate model
    NVars = 8 # Number of parameters in one individual in the population
    MaxFEval = 465 # The amount of calculation of the objective function
    globalindex = np.zeros(1)

    target_func = func_NN_calc
   

    LeftPRS  = np.array([-10.0, #tetta0
                         -10.0, #tetta1
                         -10.0, #tetta2
                         -10.0, #tetta3
                         -10.0, #tetta4                 
                         -10.0, #tetta5
                         -10.0, #tetta6
                         -10.0, #tetta7
                        ])

    RightPRS = np.array([10.0, #tetta0
                         10.0, #tetta1
                         10.0, #tetta2
                         10.0, #tetta3
                         10.0, #tetta4                 
                         10.0, #tetta5
                         10.0, #tetta6
                         10.0, #tetta7
                        ])

    ext = []
    extX = []
    extY = []
    extT = []

    globalindex[0] = 0


    NInds = NIndsMax = 15 # Number of individuals in the population    
    stepsFEval = (np.linspace(NIndsMax, MaxFEval, 16)).astype(int)    

    Pval = 0.3     
    MeanF = 0.5
    MeanCr = 0.9
    SRTC = 0.25
    ArchProb = 0.5
    SelPress = 3

    Fit2save, Fit2save_temp = np.zeros(NInds), np.zeros(NInds)
    Fitness, FitTemp = np.zeros(NInds), np.zeros(NInds)                             # Initializing empty arrays for target function values      
    PopulTemp = np.zeros((NInds,NVars))                                             # Initializing an empty array for a temporary population
    
    Popul = np.random.normal(StartPRS, (RightPRS-LeftPRS)*0.05, size=(NInds,NVars)) # Create a random population 

    # Border check. If we go out, we recreate the individual
    for i in range(NInds):
        for j in range(NVars):
            while(Popul[i][j] < LeftPRS[j] or Popul[i][j] > RightPRS[j]):
                Popul[i][j] = np.random.normal(StartPRS[j],(RightPRS[j]-LeftPRS[j])*0.05) 

    Arch = np.random.uniform(LeftPRS,RightPRS,size=(NInds,NVars)) # Create Archive

    # Border check. If we go out, we recreate the individual
    for i in range(NInds):
        for j in range(NVars):
            while(Arch[i][j] < LeftPRS[j] or Arch[i][j] > RightPRS[j]):
                Arch[i][j] = np.random.normal(StartPRS[j],(RightPRS[j]-LeftPRS[j])*0.05)

    PopArch = np.zeros((NInds,NVars))
   
    # We calculate the value of the objective function for the entire population
    for i in range(NInds):
        Fitness[i], Fit2save[i] = target_func(Popul[i:i+1,:])


    ResAll[:NInds] = Fitness

    xt = np.copy(Popul)
    yt = np.copy(Fitness).reshape(NInds)
    yt2 = np.copy(Fit2save).reshape(NInds)


    euclid_distance_archive = np.array([])

    Means = xt[np.argmin(yt)]
    Sigmas = (RightPRS-LeftPRS)*0.05

    LeftPRS_ = Means - Sigmas
    RightPRS_ = Means + Sigmas    
    LeftPRS_ = np.max((LeftPRS,LeftPRS_),axis=0)
    RightPRS_ = np.min((RightPRS,RightPRS_),axis=0)

    xlimits = np.array([LeftPRS_,RightPRS_]).T

    design_space = DesignSpace(xlimits)
    reg = KRG(design_space=design_space, print_global=False)
    reg.set_training_values(xt, yt)
    reg.train()

    globalbest = np.min(Fitness)

    while(globalindex[0] < MaxFEval):
        
        np.savetxt( os.path.join(root,"populX.txt") ,Popul)
        np.savetxt( os.path.join(root,"populY_Loss.txt") ,Fitness)
        np.savetxt( os.path.join(root,"populY_Acc.txt"), Fit2save) 
        
        indexes = np.argsort(Fitness)            
        cur_i = np.arange(0,NInds,1)
        F_DE = np.random.standard_cauchy(NInds)*0.1+MeanF
        while(any(F_DE < 0.0)):
            F_DE[np.where(F_DE < 0)] = np.random.standard_cauchy()*0.1+MeanF
        F_DE = np.clip(F_DE,0,1)
        F_DE = np.reshape(np.repeat(F_DE,NVars),(NInds,NVars))
        Cr = np.clip(np.random.normal(0,0.1,NInds) + MeanCr,0,1)            
        Cr = np.reshape(np.repeat(Cr,NVars),(NInds,NVars))                
        jrand = np.random.randint(0,NVars,NInds)
        CrChange = Cr < np.random.uniform(0,1,(NInds,NVars))
        CrChange[cur_i,jrand] = False                           
        pbest = indexes[np.random.randint(0,Pval*NInds,NInds)]                                        
        prob = np.exp(-cur_i/NInds*SelPress)
        prob /= np.sum(prob)                        
        r1_SP = np.random.choice(indexes, NInds, p=prob)
        WhereArch = ArchProb < np.random.uniform(0,1,(NInds))            
        NArch = np.sum(WhereArch)            
        r2 = np.random.randint(0,NInds,NInds)
        PopArch[WhereArch] = Arch[WhereArch]
        PopArch[np.logical_not(WhereArch)] = Popul[np.logical_not(WhereArch)]    
        PopulTemp[cur_i] = Popul[cur_i] + F_DE*(Popul[pbest] - Popul[cur_i] + Popul[r1_SP] - PopArch[r2])
        PopulTemp[CrChange] = Popul[CrChange]          
        
        for j in range(NVars):
            PopulTemp[np.where(PopulTemp[:,j] < LeftPRS[j]),j] = (Popul[np.where(PopulTemp[:,j] < LeftPRS[j]),j] + LeftPRS[j])*0.5            
            PopulTemp[np.where(PopulTemp[:,j] > RightPRS[j]),j] = (Popul[np.where(PopulTemp[:,j] > RightPRS[j]),j] + RightPRS[j])*0.5                     
        
        for i in range(NInds):
            FitTemp[i], Fit2save_temp[i] = target_func(PopulTemp[i:i+1])
            if(globalindex[0] >= MaxFEval):
                break
       
        globalbest = np.min([globalbest,np.min(FitTemp)])
        SuccessRate = np.sum(FitTemp < Fitness) / NInds     
        repl = np.zeros(NInds, dtype = 'bool')
        for i in range(NInds):
            repl[i] = FitTemp[i] < Fitness[i]
            xt = np.vstack((xt, PopulTemp[i].reshape(1, -1)))
            yt = np.hstack((yt, FitTemp[i]))
            yt2 = np.hstack((yt2, Fit2save_temp[i]))
        rand_i = np.random.randint(0,NInds,np.sum(repl)) 
        Arch[rand_i] = Popul[repl]
        Popul[repl] = PopulTemp[repl]            
        Fitness[repl] = FitTemp[repl]
        Fit2save[repl] = Fit2save_temp[repl]

        MeanF = np.power(SuccessRate,SRTC)
          
        besti = np.argmin(Fitness)
        print(Fitness[besti],Popul[besti])
        if(globalindex[0] >= MaxFEval):
            break

        for i in range(n_sur):
            point_sur, fit_sur, fit2save_aquis  = Aquisition_LSRDE(reg,LeftPRS_,RightPRS_)
            print("point_sur, fit_sur",point_sur, fit_sur)

            xt = np.vstack((xt, point_sur.reshape(1, -1)))
            yt = np.hstack((yt, fit_sur))
            yt2 = np.hstack((yt2, fit2save_aquis))
            print("xt.shape",xt.shape)

            worst_idx = np.argmax(Fitness)
            if(Fitness[worst_idx] > fit_sur):
                print("replace",Fitness[worst_idx],fit_sur)
                Popul[worst_idx] = point_sur
                Fitness[worst_idx] = fit_sur
                Fit2save[worst_idx] = fit2save_aquis
      
        Means = xt[np.argmin(yt)]
        print("Means",Means)

        LeftPRS_ = Means - Sigmas
        RightPRS_ = Means + Sigmas
        LeftPRS_ = np.max((LeftPRS,LeftPRS_),axis=0)
        RightPRS_ = np.min((RightPRS,RightPRS_),axis=0)
        xlimits = np.array([LeftPRS_,RightPRS_]).T
        print("xlimits",xlimits)
        design_space = DesignSpace(xlimits)
        reg = KRG(design_space=design_space, print_global=False)
        reg.set_training_values(xt, yt)
        reg.train()     
        
        np.savetxt( os.path.join(root,"xt.txt"), xt)
        np.savetxt( os.path.join(root,"yt_loss.txt"), yt)
        np.savetxt( os.path.join(root,"yt_acc.txt"), yt2)

        euclid_distance_archive = np.append(euclid_distance_archive, euqlid_dist(Popul)) # We count the distance and save it
        np.savetxt( os.path.join(root,"euclid_dist.txt"), euclid_distance_archive)

        print(np.argmin(yt),np.min(yt))


            
    end_timer = time.time()
    print(f"\nTime: %.2f" % ((end_timer - start_timer)/60)/60,"hours")
    
    """SAVING"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
