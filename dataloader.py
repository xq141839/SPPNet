import os
from skimage import io, transform, color,img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensor
from point import RandomExtractor, CentreExtractor, CNPS
import torchvision.transforms as transforms
import torch.nn.functional as F

class sam_inputer(Dataset):
        def __init__(self,path,data, transform=None, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = path
            self.folders = data
            self.transforms = transform
            self.to_tesnor = transforms.Compose([transforms.ToTensor(), ])
            self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.pixel_std = torch.Tensor(pixel_mean).view(-1, 1, 1)
            self.img_size = 1024
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_id = list(self.folders[idx].split('.'))[0]
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',image_id)
            npy_path = os.path.join(self.path,'npy/',image_id) + '.npy'

            point_coord, point_class = CNPS(npy_path)
            point_coord = torch.tensor(point_coord)
            point_class = torch.tensor(point_class)
    
            img = io.imread(image_path)[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True)
            
            img_vit = self.to_tesnor(img)
            img_vit, h, w = self.preprocess(img_vit)

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
   
            return (img, point_coord, point_class, img_vit, mask, image_id, h, w)

        def preprocess(self, x):
            """Normalize pixel values and pad to a square input."""
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))

            return x, h, w

# class sam_inputer(Dataset):
#         def __init__(self,path,data, transform=None):
#             self.path = path
#             self.folders = data
#             self.transforms = transform
#             self.to_tesnor = transforms.Compose([transforms.ToTensor(), ])
#             self.pixel_mean = (123.675, 116.280, 103.530)
#             self.pixel_std = (58.395, 57.12, 57.375)
#             self.img_size = 1024
        
#         def __len__(self):
#             return len(self.folders)
              
        
#         def __getitem__(self,idx):
#             image_id = list(self.folders[idx].split('.'))[0]
#             image_path = os.path.join(self.path,'images/',self.folders[idx])
#             pt_path = os.path.join(self.path,'features/',image_id)
#             mask_path = os.path.join(self.path,'masks/',image_id)
#             npy_path = os.path.join(self.path,'npy/',image_id) + '.npy'

#             point_coord, point_class = CNPS(npy_path)
#             point_coord = torch.tensor(point_coord)
#             point_class = torch.tensor(point_class)
    
#             img = io.imread(image_path)[:,:,:3].astype('float32')
#             mask = io.imread(mask_path+'.png', as_gray=True)
            

#             self.pixel_mean = np.array(self.pixel_mean, dtype=np.float32)
#             self.pixel_std = np.array(self.pixel_std, dtype=np.float32)
#             img_vit = (img - self.pixel_mean) / self.pixel_std 

#             h, w = img_vit.shape[:-1]
#             padh = self.img_size - h
#             padw = self.img_size - w
#             img_vit = self.to_tesnor(img_vit)
#             img_vit = F.pad(img_vit, (0, padw, 0, padh))

#             augmented = self.transforms(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']
   
#             # return (img, sam_feature, mask, image_id)
#             return (img, point_coord, point_class, img_vit, mask, image_id, h, w)
