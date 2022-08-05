#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INitializes the model and loss functions for the different AV correlation models
"""
import sys
import torch

import losses

def getLossFn(config):
    
         """ Get loss function """
         
         loss_fn = None
         
         if config.loss.endswith("loss_ntxent"):
             print("NTXent Contrastive loss")
             loss_fn = losses.NTXentLoss(config)     
         
         return loss_fn

def getLoss(config, loss_fn, device, logits, vemb, aemb, y_lab, av_dist=None, neg_aemb=None):
      """ Return the batch loss using the specified loss function """
      
      batch_loss = None
      
      if config.loss.endswith("loss_ntxent"):   #NT cross entropy loss
         batch_loss = loss_fn(logits, y_lab, device) #vis and aud normalized projected vectors     
      return batch_loss        


     
def getModel(config):
    
         norm = config.norm 
         if config.model == "av_contrast_simclr":
            import models_simclr as avm
            #pairwise A-V similarity based on SimCLR model with attention (contrastive learning with negatives from same batch) 
            model = avm.AVSimCLR(config, norm)   
         else:
           print("unknown model %s"%config.model)
           sys.exit()
           
         return model
                 
        
def initModel(config):
         """ initialize model and loss function as per config """
         
         loss_fn = getLossFn(config)     
         model = getModel(config)
         
         return model, loss_fn
     
