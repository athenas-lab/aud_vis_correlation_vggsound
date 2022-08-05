import os
import sys
import json

from torch.utils.data import DataLoader
from torchvision import transforms, utils
import vggsound_dataset_contrast as avd

""" 
Data loader for self-supervised contrastive learning approach using vggsound dataset
"""
   
class VideoNoisyAudLoader():  
   
   def __init__(self, conf):
        
        self.conf = conf
      
   def getDataLoader(self,  dsets, mode):
        """ Return data loader """

        #get the list of data 
        dlist = {"train": "lists/vid2aud_train.json",
                "val": "lists/vid2aud_val.json",
                "test": "lists/vid2aud_test.json",
                }
      
        #get the loader for self-supervised learning
        if self.conf.getVisType() == "raw" and self.conf.getVisRate() == 1:
            trans_dict = getFrameTransforms(self.conf.img_size)
            data = avd.VGGSound(self.conf, dlist[mode], vis_transform=trans_dict[mode])

        do_shuffle = True if mode == "train" else False #dont shuffle test  data
        loader = DataLoader(data, batch_size=self.conf.batch_size, shuffle= do_shuffle, num_workers=self.conf.nworkers)
        return loader 
    


    
def getFrameTransforms(inp_size):
        data_transforms = {
          #data augmentation and normalization for train set
          "train": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.RandomResizedCrop(inp_size),
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     #where are these numbers coming from? Is this required?
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ]),
          #data normalization for val and test set. No augmentation
          "val": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.CenterCrop(inp_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ]),
          "test": transforms.Compose([
                     transforms.Resize((inp_size, inp_size)),
                     #transforms.CenterCrop(inp_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])
          }
        return data_transforms
    
