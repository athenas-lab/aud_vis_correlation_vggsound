#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Convolutional blocks for stacking """

def getNonLinear(nlu):

    if nlu == "relu":
       return nn.ReLU(inplace=False)
    elif nlu == "lrelu": #leaky RelU
        return nn.LeakyReLU(inplace=False) #(slope, inplace)
    elif nlu == "celu": #continuously exponential RelU
        return nn.CELU(inplace=False)
    elif nlu == "prelu": #parametric ReLU #dont use weight decay when using this,
        return nn.PReLU(num_parameters=1)
    elif nlu == "glu": #gated linear unit
       return F.glu
    else:
       return None
   
def getNorm(norm):

    if norm =="bn": #default batchnorm
       return nn.BatchNorm2d
    else:
       return None
   
def Conv3x3(in_ch, out_ch, stride):
    """ 3x3 convolution with padding """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, bias=True)
    #conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, 
    #                 padding=1, groups=groups, bias=False)
    return conv

def Conv1x1(in_ch, out_ch, stride=1):
    """ 1x1 convolution """
    conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False)
    return conv    

def Conv1D(in_ch, out_ch, stride=1):
    """ 1D convolution """
    conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=stride, bias=True)
    return conv    

class convBlock(nn.Module):
  """ Basic block: [(3x3, out_ch) ; (3x3, out) conv + BN + ReLU] """
  
  def __init__(self, in_ch, out_ch, stride, norm, num_conv):
    
      super(convBlock, self).__init__()

      norm_fn = getNorm(norm)  
      layers = []

      for i in range(num_conv):
          if i == 0:
             layers.append(Conv3x3(in_ch, out_ch, stride=stride))
          else:
             #layers.append(Conv3x3(out_ch, out_ch, stride=stride)) 
             layers.append(Conv3x3(out_ch, out_ch, stride=1)) 
          if norm_fn is not None:    
              layers.append(norm_fn(out_ch, affine=True, track_running_stats=True))
          layers.append(getNonLinear("relu"))           
                     
      self.op = nn.Sequential(*layers)    

  def forward(self, x):
    
      out = self.op(x)
      return out
  
    
class projectBlock(nn.Module):

      def __init__(self, in_ch, out_ch):
    
          super(projectBlock, self).__init__() 
          """ 1x1 convolution """
          self.op = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1, bias=False)
          
          
      def forward(self, x):
    
          out = self.op(x)
          return out     
      
class linearAttBlock(nn.Module):

      def __init__(self, in_ch, norm_att=True):
    
          super(linearAttBlock, self).__init__() 
          """ 1x1 convolution """
          self.norm_att = norm_att
          self.op = nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=1, padding=0, stride=1, bias=False)
          
          
      def forward(self, loc, gl):
          
          N,C,W,H = loc.shape
          cf = self.getCompatFn(loc, gl)
          
          #normalize the attention scores
          if self.norm_att:
              att = F.softmax(cf.view(N, 1, -1), dim=2).view(N, 1, W, H)
          else:
              att = torch.sigmoid(cf)
              
          #weigh the local feature vectors with attention weights    
          g = torch.mul(att.expand_as(loc), loc)  
          
          if self.norm_att:
              g = g.view(N, C,-1).sum(dim=2) # [N, C]
          else:
              g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C) 
              
          return cf.view(N, 1, W, H), g           
      
      def getCompatFn(self, loc, gl):
          
          compat_fn = "add"
          if compat_fn == "dot_prod": 
              cf = torch.dot(loc, gl)
          else:
              cf = self.op(loc+gl)
              
          return cf    
      
        
