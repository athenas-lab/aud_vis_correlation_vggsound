#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
   Utilities to log and display attention maps 
   Adapted from https://github.com/SaoYan/LearnToPayAttention/blob/master/train.py 
"""
import cv2
import numpy as np

import torch
import torchvision.utils as utils
import torch.nn.functional as F 

min_factor_vis = {
        "VisNet": [None,None,None],
        "VisNetAtt3BeforePool1": [1, 2, 4],
        "VisNetAtt3AfterPool1":  [2, 4, 8],
        "VisNetAtt2AfterPool1":  [2, 4, None],
        "VisNetAtt2AfterPool2":  [4, 8, None],
        "VisNetAtt1AfterPool1": [4,None,None],
        "VisNetAtt1AfterPool2": [8,None,None]    
        }

min_factor_aud = {
        "AudNet": [None,None,None],
        "AudNetAtt3BeforePool1": [2, 4, None],
        "AudNetAtt3BeforePool2": [1, 2, 4],
        "AudNetAtt3AfterPool1":  [2, 4, None],
        "AudNetAtt2AfterPool1":  [2, 4, None],
        "AudNetAtt1AfterPool1":  [4,None,None]
        }

def visualizeAtt(model, epoch, writer, images, grid, dtype, tr_test):
       """ visualize attention maps for train or test data """
       
       norm_att = True
       if dtype == "aud":
          _, local_att, _ = model.module.afeatures(images)  
          if local_att is None:
             return 
          c1, c2, c3 = local_att  
          min_up_factor = min_factor_aud[model.module.afeatures.__class__.__name__]
          #min_up_factor = min_factor_aud[model.module.aud_name]
          #c3 = None #for audio,  the 3rd attention layer does not have same pooling size along both dimensions, so we disable the scale up
       else: #visual data
          _, local_att, _ = model.module.vfeatures(images)  
          if local_att is None:
              return
          c1, c2, c3 = local_att
          min_up_factor = min_factor_vis[model.module.vfeatures.__class__.__name__] 
          #min_up_factor = min_factor_vis[model.module.vis_name] 
       
       #log attention maps
       if (c1 is not None) and (min_up_factor[0] is not None):
          #print("train/c1")
          att1 = compAttMap(grid, c1, min_up_factor[0], norm_att) 
          writer.add_image("%s_%s/att_map_1"%(tr_test,dtype), att1, epoch)  
       if (c2 is not None) and (min_up_factor[1] is not None):
          #print("train/c2")
          att2 = compAttMap(grid, c2, min_up_factor[1], norm_att) 
          writer.add_image("%s_%s/att_map_2"%(tr_test, dtype), att2, epoch)     
       if (c3 is not None) and (min_up_factor[2] is not None):
          #print("train/c3") 
          att3 = compAttMap(grid, c3, min_up_factor[2], norm_att) 
          writer.add_image("%s_%s/att_map_3"%(tr_test, dtype), att3, epoch)
       return
   
def visualizeAttMaps(model, epoch, writer, images, I_train, I_test, dtype):
       """ 
        Visualize the attention maps for train and test data  
       """
#   
       #log attention maps for the training data 
       visualizeAtt(model, epoch, writer, images[0], I_train, dtype, "train")
       #log attention maps for the test data 
       visualizeAtt(model, epoch, writer, images[1], I_test, dtype, "test")
       return
   
    
def compAttMap(Igrid, c, up_factor, norm_att=True, nrow=4):
    
    #convert torch img tensor dims [C,W,H] --> [W,H,C] and use numpy format
    img = Igrid.permute(1,2,0).cpu().numpy()
    
    #compute the heatmap
    if norm_att: #normalized attention uses softmax
       N, C, W, H = c.size()
       hm = F.softmax(c.view((N,C,-1)), dim=2).view(N, C, W, H)
       if up_factor > 1:
          hm = F.interpolate(hm, scale_factor= up_factor, mode="bilinear", align_corners=False)
       attn = utils.make_grid(hm, nrow=nrow, normalize=True, scale_each=True)
    else:
       hm = F.sigmoid(c)        
       if up_factor > 1:
          hm = F.interpolate(hm, scale_factor= up_factor, mode="bilinear", align_corners=False)
       attn = utils.make_grid(hm, nrow=nrow, normalize=False)      
   
    #convert heatmap to pixel values
    attn =  attn.permute((1,2,0)).mul(255).byte().cpu().numpy()   
    attn =  cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn =  cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn)/255 #normalize
    #print(c.size())
    #print("(image, attn):", img.shape, attn.shape)
    #add heatmap to the raw image
    vis = 0.6*img + 0.4 * attn
       
    vis = torch.from_numpy(vis).permute(2,0,1)
    return vis
