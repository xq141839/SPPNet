import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import sam_inputer
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import sppnet
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

def postprocess_masks(
        masks: torch.Tensor,
        input_size: (Tuple[int, ...]),
        original_size: (Tuple[int, ...]),
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='monuseg/images/',type=str, help='the path of dataset')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--model',default='save_models/model_best.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs('debug/',exist_ok=True)

    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    test_dataset = sam_inputer(args.dataset,test_files, get_transform())

    model = sppnet.Model()
    model.load_state_dict(torch.load(args.model))

    model = model.cuda()
    
    acc_eval = Accuracy()
    pre_eval = Precision()
    dice_eval = Dice()
    recall_eval = Recall()
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    id_lists = []
    
    since = time.time()
    if args.debug:
        for image_id in test_files:
            img = cv2.imread(f'{args.dataset}images/{image_id}')
            img = cv2.resize(img, ((256,256)))
            img_id = list(image_id.split('.'))[0]
            cv2.imwrite(f'debug/{img_id}.png',img)
    
    with torch.no_grad():
        for img, point_coord, point_class, img_vit, mask, img_id, h, w in tqdm(test_dataset):

            point_coord = Variable(torch.unsqueeze(point_coord, dim=0), requires_grad=False).cuda()
            point_class = Variable(torch.unsqueeze(point_class, dim=0), requires_grad=False).cuda() 
            img_vit = Variable(torch.unsqueeze(img_vit, dim=0), requires_grad=False).cuda()
            img = Variable(torch.unsqueeze(img, dim=0), requires_grad=False).cuda()            
            mask = Variable(torch.unsqueeze(mask, dim=0), requires_grad=False).cuda()

            torch.cuda.synchronize()
            start = time.time()
            pred = model(img, point_coord, point_class, img_vit, (h, w))
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)

            pred_tmp = postprocess_masks(pred, (1000, 1000), (1000, 1000))
            # print(pred_tmp)

            pred = torch.sigmoid(pred)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            

            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()
            
            
            if args.debug:
                img_id = list(img_id.split('.'))[0]
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy==1] = 255 
                cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy)
                
                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy==1] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png',mask_numpy)
            iouscore = iou_eval(pred,mask)
            dicescore = dice_eval(pred,mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
     
            accscore = acc_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            id_lists.append(img_id)
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since

    result_dict = {'image_id':id_lists, 'miou':iou_score, 'dice':dice_score}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('best.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))
    print('mean accuracy:',round(np.mean(acc_score),4),round(np.std(acc_score),4))
    print('mean precsion:',round(np.mean(pre_score),4),round(np.std(pre_score),4))
    print('mean recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('mean F1-score:',round(np.mean(f1_score),4),round(np.std(f1_score),4))
