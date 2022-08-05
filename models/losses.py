#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different loss functions for training AV correlation models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCELoss(nn.Module):
    """
    BCEWithLogits  loss     
    """
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        
    def forward(self, label, logits, device):
         
        #BCE loss
        y_lab = torch.nn.functional.one_hot(label, num_classes=logits.size(1)).type(torch.FloatTensor).to(device)

        loss = self.bce(logits, y_lab)        
        #print("bce loss", label, loss)

        return loss
    
class NTXentLoss(nn.Module):
    """
    Normalized temperature cross-entropy loss as defined in SimCLR paper + InfoNCELoss as defined in the CPC paper. 
    """
    def __init__(self, cfg=None):
        super(NTXentLoss, self).__init__()
        print("nt_xent_loss")
        if cfg is not None:
            self.temp = cfg.margin            
            self.pos_den = cfg.cont_cfg["pos_denom"]
            use_cosine = cfg.cont_cfg["use_cosine"]            
        else: #default 
            self.temp = 0.5
            self.pos_den = False #simclr loss
            use_cosine = True
            
        if use_cosine:
           self.sim_fn = self._cosineSim
        else:
           self.sim_fn = self._dotSim
           
        if self.pos_den:#cross entropy loss if positive sample is included in the denominator
           self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum") 
    
    @staticmethod    
    def getMask(N):
        diag = np.eye(N)
        mask1 = torch.from_numpy((diag))
        mask = (1- mask1).type(torch.bool)
        #print("diag={}, \n mask1={}, \n mask={}".format(diag, mask1, mask))
        return mask
    
    @staticmethod    
    def getMask2(N):
        diag = np.eye(2*N)
        l1 = np.eye(2*N, 2*N, k=-N)
        l2 = np.eye(2*N, 2*N, k=N)
        mask1 = torch.from_numpy((diag + l1 + l2))
        mask = (1- mask1).type(torch.bool)
        print("diag={}, \n l1={}, \n l2={}, \n mask1={}, \n mask={}".format(diag, l1, l2, mask1, mask))
        return mask
    
    @staticmethod        
    def  _dotSim(x, y):
        #compute the similarity of 2 input vectors as a dot product
        x1 = x.unsqueeze(1) #(N,1,C)
        y1 = y.T.unsqueeze(0) #(1,C,N)
        s = torch.tensordot(x1, y1, dim=2) #(N,N)
        return s
    
    @staticmethod        
    def  _cosineSim(x, y):
        #compute the similarity of 2 input vectors as a dot product
        x1 = x.unsqueeze(1) #(N,1,C)
        y1 = y.unsqueeze(0) #(1,N,C)
        sim = torch.nn.CosineSimilarity(dim=-1) 
        s = sim(x1, y1) #(N,N)
        #print(x1.shape, y1.shape, s)
        return s
        
    def compSimclr(self, vz, az, device):
        #print("===bolts===")
        bs = vz.shape[0] #batch size
        cov = torch.mm(vz, az.t()) 
        #print("sim_matrix", cov)
        sim = torch.exp(cov/self.temp)
        
        #mask to extract negative (off diagonal elements from a matrix)
        mask = ~torch.eye(bs, device=device).bool()
        den = sim.masked_select(mask).view(bs,-1).sum(dim=-1)
        #print("den", den)
        #pos sim 
        num = torch.exp(torch.sum(vz*az, dim=-1)/self.temp)
        #print("num", num)
    
        loss = -torch.log(num/den)
        loss = loss.mean()
        #print("=====avg loss", loss)
        return loss
        
    def forward(self, vz, az, device, red_avg = True):
        
        bs = vz.shape[0] #batch size
        mask = self.getMask(bs) #mask to filter out diagonal elements (positive pairs)
        mask = mask.to(device).type(torch.bool)

        #similarity between every vis latent vec and every aud latent vector in the batch
        sim_matrix = self.sim_fn(vz, az) #(NxN)
        #print("sim_matrix", sim_matrix)
        #diagonal entries are the positive pairs. Extract the sim values for pos pairs
        pos = torch.diag(sim_matrix)
        pos = pos.view(bs, 1)
        #negative samples are the off-diagonal entries
        neg = sim_matrix[mask].view(bs, -1)
        #print("pos={}, neg={}".format(pos, neg))
       
        if self.pos_den: #cross entropy loss
            
            #[sim values for the pos pair, sim vals for neg pair concatenated columnwise]
            #first column corresponds to positive similarity, remaininng N-1 columns are sim values for neg pairs
            logits = torch.cat((pos, neg), dim=1)  
            logits /= self.temp #normalize the logits with the temp value
            
            #Gt is 0=> for each sample, the first column indicates the positive sample
            labels = torch.zeros(bs).to(device).long()
            loss = self.loss_fn(logits, labels) #cross entropy loss
            #print("logits={}, loss={}".format(logits, loss))
            if red_avg:
               loss /= bs #avg loss
        else: #simclr loss  without pos element in denominator
            loss = self.compSimclr(vz, az, device)
        #print("avg loss", loss)   
        return loss  
    
#    def forward_pos_den(self, vz, az, device, red_avg = True):
        
#        bs = vz.shape[0] #batch size
#        mask = self.getMask(bs) #mask to filter out diagonal elements (positive pairs)
#        mask = mask.to(device).type(torch.bool)

        #similarity between every vis latent vec and every aud latent vector in the batch
#        sim_matrix = self.sim_fn(vz, az) #(NxN)
        #print("sim_matrix", sim_matrix)
        #diagonal entries are the positive pairs. Extract the sim values for pos pairs
#        pos = torch.diag(sim_matrix)
#        pos = pos.view(bs, 1)
#        #negative samples are the off-diagonal entries
#        neg = sim_matrix[mask].view(bs, -1)
        #print("pos={}, neg={}".format(pos, neg))
       
        #[sim values for the pos pair, sim vals for neg pair concatenated columnwise]
        #first column corresponds to positive similarity, remaininng N-1 columns are sim values for neg pairs
#        logits = torch.cat((pos, neg), dim=1)  
#        logits /= self.temp #normalize the logits with the temp value
        
        #Gt is 0=> for each sample, the first column indicates the positive sample
#        labels = torch.zeros(bs).to(device).long()
#        loss = self.loss_fn(logits, labels) #cross entropy loss
        #print("logits={}, loss={}".format(logits, loss))
#        if red_avg:
#           loss /= bs #avg loss
        #print("avg loss", loss)   
#        return loss
    
    
    
