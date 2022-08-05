#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines dataset classes for recommending sounds correlated with video frames """

import os
import sys
import logging
import re
import json

#cv2 and skimage do not return Image in a format that tranforms likes
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms



#root folder for vgg visual and audio data
vis_root = "data/vggsound/frames_1fps"

class ImgDataset(Dataset):
    
    """ 
       Data loader for a list of images or video frames. 
    """
    
    def __init__(self, conf, data_list, vis_transform):
        self.conf = conf
        self.data_list = data_list[0]
        self.labels = data_list[1]
        self.vis_transform = vis_transform
        print("loaded meta-data:{}".format(len(self.data_list)))
        logging.info("loaded meta-data:{}".format(len(self.data_list))) 
        np.random.seed(144)
    
    def __len__(self):
       """ return length of the dataset """
      
       return len(self.data_list)
     
    def __getitem__(self, n):
       """ Return a single visual frame/image and corresponding file name   
                     
       """  
       
       img_path = self.data_list[n]
       x = self.getVisFrame(img_path)
      
       #pytorch dataloader does not support None value (sample becomes None). 
       #So we remove the negative label keys if they are not required.
       sample = {"v": x, "v_fn": img_path, "label": self.labels[n]}  
          
       #print("loaded data shapes:", x.shape, sample["v_fn"])
              
       return sample
  
    def globalContrastNorm(self, img):
        """ global contrast pixel normalization """

        eps = 1e-09
        lmda = 5
        s = 1
        img_res = img.resize((224, 224))
        x = np.array(img_res)
        x_avg = np.mean(x)

        x_new = x - x_avg
        contrast = np.sqrt(lmda + np.mean(x_new**2))
        new_img = s*x_new/ max(contrast, eps)
        new_img = np.transpose(new_img, (0,2,1))
        new_img = np.transpose(new_img, (1,0,2))
        new_img = torch.from_numpy(new_img).float()
        return new_img


    def getVisFrame(self, img_path):
       #return video frame given the path 
        
       ip = os.path.join(vis_root, img_path) 
       x = Image.open(ip)
       #print(img_path, "original shape=", x.shape)

       #x = self.globalContrastNorm(x)

       #apply data transform if required
       if self.vis_transform is not None:
          #print(x.size)
          x = self.vis_transform(x)
       return x   



""" Data loaders to load test data """

#def getAudLoader(conf, data_list):    
#
#    data = dc.AudDataset(conf, data_list)
#    dl = DataLoader(data, batch_size=conf.batch_size, shuffle= False, num_workers=conf.nworkers)
#    return dl    

def getImgLoader(conf, data_list):    
    trans_dict = getFrameTransforms(conf.img_size)  
    data = ImgDataset(conf, data_list, trans_dict["test"])
    dl = DataLoader(data, batch_size=conf.batch_size, shuffle= False, num_workers=conf.nworkers)
    return dl    


def getFrameTransforms(inp_size):
        data_transforms = {
          #data augmentation and normalization for train set
          "train": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.Resize((720, 1280)),
                     #transforms.RandomResizedCrop(inp_size),
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     #where are these numbers coming from? Is this required?
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ]),
          #data normalization for val and test set. No augmentation
          "val": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.Resize((720, 1280)),
                     #transforms.CenterCrop(inp_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
          "test": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.Resize((720, 1280)),
                     #transforms.CenterCrop(inp_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])
          }
        return data_transforms
