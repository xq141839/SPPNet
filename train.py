import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import sam_inputer
from sklearn.model_selection import GroupKFold
from loss import *
from tqdm import tqdm
import json
import sppnet
from modeling.tiny_vit_sam import TinyViT


def get_train_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        # A.HorizontalFlip(p=0.25),
        # A.RandomBrightness(p=0.25),
        # A.ShiftScaleRotate(shift_limit=0,p=0.25),
        # A.CoarseDropout(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])

def get_valid_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss = []
            running_corrects = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for img, point_coord, point_class, img_vit, labels, _, h, w in tqdm(dataloaders[phase]):      
                # wrap them in Variable
                if torch.cuda.is_available():

                    point_coord = Variable(point_coord.cuda())
                    point_class = Variable(point_class.cuda())
                    img_vit = Variable(img_vit.cuda())
                    img = Variable(img.cuda())
                    labels = Variable(labels.cuda())
                    #label_for_ce = Variable(label_for_ce.cuda())
                else:
                    img, point_coord, point_class, img_vit, labels = Variable(img), Variable(point_coord), Variable(point_class), Variable(img_vit), Variable(labels)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                #label_for_ce = label_for_ce.long()
                # forward
                outputs = model(img, point_coord, point_class, img_vit, (h[0].item(), w[0].item()))
                # print(outputs)

                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs,labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects.append(score.item())
             

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)
            
            print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'train':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    
    torch.save(best_model_wts, 'save_models/model_best.pth')

    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='monuseg/images', help='the path of images')
    parser.add_argument('--prompt', type=str,default='sam_vit_h_4b8939.pth', help='')
    parser.add_argument('--encoder', type=str,default='mobile_sam.pt', help='')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'save_models/',exist_ok=True)
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    val_files = df['valid']
    train_files = df['train']
    
    train_dataset = sam_inputer(args.dataset,train_files,get_train_transform())
    val_dataset = sam_inputer(args.dataset,val_files,get_valid_transform())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1 ,drop_last=True)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
   
    vit_encoder =  TinyViT(img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=True,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )

    model_ft = sppnet.Model(image_encoder=vit_encoder)

    encoder_dict = torch.load(args.encoder)
    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model_ft.load_state_dict(pre_dict, strict=False)

    prompt_dict = torch.load(args.prompt)
    pre_dict = {k: v for k, v in prompt_dict.items() if list(k.split('.'))[0] != 'image_encoder'}
    model_ft.load_state_dict(pre_dict, strict=False)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
        
    # Loss, IoU and Optimizer
    if args.loss == 'ce':
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss()
    
    accuracy_metric = IoU()
    optimizer_ft = optim.Adam(model_ft.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.8)
    #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=5, factor=0.1,min_lr=1e-6)
    Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')

