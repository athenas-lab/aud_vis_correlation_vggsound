#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbone models for AV contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import convBlock, linearAttBlock, projectBlock, getNonLinear
from blocks import getNorm, Conv3x3

"""==========================VGG models ===================="""
class BasicBlock(nn.Module):
  """ Basic block: [(3x3, out_ch) ; (3x3, out) conv + BN + ReLU] """
  
  def __init__(self, in_ch, out_ch, stride, ps, norm):
    
      super(BasicBlock, self).__init__()

      self.conv1 = Conv3x3(in_ch, out_ch, stride)
      self.conv2 = Conv3x3(out_ch, out_ch, stride=1)
      self.norm_layer = norm
      if self.norm_layer is not None:
          norm_fn = getNorm(norm)  
          self.bn1 = norm_fn(out_ch)
          self.bn2 = norm_fn(out_ch)

      self.relu = getNonLinear("relu")

  def forward(self, x):
    
      out = self.conv1(x)
      if self.norm_layer:
          out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      if self.norm_layer:
          out = self.bn2(out)
      out = self.relu(out)

      return out
  
class AudNet(nn.Module):
  """ Audio subnet """

  def __init__(self, norm, init_wts=True):

      super(AudNet, self).__init__()

      #expected input shape: [batch_size x 1 x num_frames x feature_dim]
      out_ch = [64, 128, 256, 512]
      self.c1 = BasicBlock(in_ch=1, out_ch=out_ch[0], stride=1, ps=2, norm=norm) #for log-mel spect features
      #self.c1 = BasicBlock(in_ch=1, out_ch=out_ch[0], stride=2, ps=2) #for spectrogram features as on Objects that Sound paper
      self.c2 = BasicBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, ps=2, norm=norm)
      self.c3 = BasicBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, ps=2, norm=norm)
      self.c4 = BasicBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, ps=2, norm=norm)
      self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool_final = nn.MaxPool2d(kernel_size=(16, 12), stride=(16, 12))


      fc_dim = 128
      self.embed = nn.Sequential(
                nn.Linear(512, fc_dim),
                getNonLinear("relu"),
                #nn.Dropout(p=0.5),
                nn.Linear(fc_dim, fc_dim)
                )
      self.norm = F.normalize
      
      
  def forward(self, x):
     
      #conv features
      out = self.c1(x)
#      print("c1 features", out.shape)  #bx64*64*100
      out = self.pool1(out)
#      print("p1 features", out.shape)  #bx64*64*50
      out = self.c2(out)
#      print("c2 features", out.shape)  #bx128x64*50
      out = self.pool(out)
#      print("p2 features", out.shape)  #bx128x64*25
      out = self.c3(out)
#      print("c3 features", out.shape)  #bx256x32x25  
      out = self.pool(out)
#      print("p3 features", out.shape)  #bx256x16x12
      out = self.c4(out)
#      print("c4 features", out.shape)  #bx512x16x12
      out = self.pool_final(out)
#      print("aud features:", out.shape) #bx512x1x1
      
      
      ##out = self.features(x)
      
      out = out.view(out.shape[0], -1)
#      print("flatten:", out.shape)     #bx512

      #conv embeddings
      out = self.embed(out)
#      print("embed:", out.shape)       #bx128

      return out

     

class VisNet(nn.Module):
  """ Visual subnet """

  def __init__(self, norm, init_wts=True):

      super(VisNet, self).__init__()

      #expected input shape: [batch_size x 3 x 224 x 224]
      out_ch = [64, 128, 256, 512]
      self.c1 = BasicBlock(in_ch=3, out_ch=out_ch[0], stride=2, ps=2, norm=norm)
      self.c2 = BasicBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, ps=2, norm=norm )
      self.c3 = BasicBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, ps=2, norm=norm)
      self.c4 = BasicBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, ps=2, norm=norm)

      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool_final = nn.MaxPool2d(kernel_size=14, stride=14)

      fc_dim = 128
      self.embed = nn.Sequential(
                nn.Linear(512, fc_dim),
                getNonLinear("relu"),
                #nn.Dropout(p=0.5),
                nn.Linear(fc_dim, fc_dim)
                )
      self.norm = F.normalize
      
      
  def forward(self, x):
      
      #conv features
      out = self.c1(x)
      #print("c1 features", out.shape) #[128, 64, 112, 112]
      out = self.pool(out)
      #print("p1 features", out.shape) #[128, 64, 56, 56]
      out = self.c2(out)
      #print("c2 features", out.shape) #[128, 128, 56, 56]
      out = self.pool(out)
      #print("p2 features", out.shape) #[128, 128, 28, 28]
      out = self.c3(out)
      #print("c3 features", out.shape) #[128, 256, 28, 28]
      out = self.pool(out)
      #print("p3 features", out.shape) #[128, 256, 14, 14]
      out = self.c4(out)
      #print("c4 features", out.shape) #[128, 512, 14, 14]
      out = self.pool_final(out)
      #print("vis features:", out.shape) #[128, 512, 1, 1]
      
      ##out = self.features(x)
     
      out = out.view(out.shape[0], -1)  #[128, 512]
      #print("flatten:", out.shape)

      out = self.embed(out)            #[128, 128]
      #print("embed:", out.shape)
      
      return out


"""==========================Attention-based VGG models ===================="""
class AudNetAtt3BeforePool2(nn.Module):
  """ Audio subnet """
  """ Implements the self attention model described in Learn to Pay Attention """  
  def __init__(self, norm, init_wts="xav_uni", normalize_att = True):

      super(AudNetAtt3BeforePool2, self).__init__()
      print(self.__class__.__name__)
      self.att =True
      
      #expected input shape: [batch_size x 3 x 224 x 224]
      out_ch = [64, 128, 256, 512, 512, 512, 512]
      self.c1 = convBlock(in_ch=1, out_ch=out_ch[0], stride=1, norm=norm, num_conv=2) #3->64
      self.c2 = convBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, norm=norm, num_conv=2) #64->128
      self.c3 = convBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, norm=norm, num_conv=3) #128->256
      self.c4 = convBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, norm=norm, num_conv=3) #256->512
      self.c5 = convBlock(in_ch=out_ch[3], out_ch=out_ch[4], stride=1, norm=norm, num_conv=3) #512->512
      self.c6 = convBlock(in_ch=out_ch[4], out_ch=out_ch[5], stride=1, norm=norm, num_conv=1) #512->512
      self.c7 = convBlock(in_ch=out_ch[5], out_ch=out_ch[5], stride=1, norm=norm, num_conv=1) #512->512
      
      
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
      #self.pool_final = nn.MaxPool2d(kernel_size=14, stride=14)

      if self.att:
          self.project = projectBlock(out_ch[2], out_ch[6])    #[256, 512] project the global feature dim onto local feat dim
          self.att1 = linearAttBlock(out_ch[6], norm_att=True)
          self.att2 = linearAttBlock(out_ch[6], norm_att=True)
          self.att3 = linearAttBlock(out_ch[6], norm_att=True)
          
      emb_dim = 128
      
      self.dense = nn.Conv2d(in_channels=out_ch[5], out_channels=out_ch[5], kernel_size=(4, 3), stride=1, padding=0, bias=True)
      
      if self.att:
         glob_dim = out_ch[6] *3
      else:
          glob_dim = out_ch[6]
         
      self.embed =  nn.Sequential(
                      nn.Linear(glob_dim, emb_dim, bias=True),
                      getNonLinear("relu")       
                    )
      
      self.norm = F.normalize
      
      
  def forward(self, x):
      
      #conv features
      out = self.c1(x)
      #print("c1 features", out.shape) #[128, 64, 64, 100]
      out = self.c2(out)
      #print("c2 features", out.shape) #[128, 128, 64, 100]
      local1 = self.c3(out)
      #print("c3 features", local1.shape) #[128, 256, 64, 100]
      out = self.pool(local1)
      #print("local1 features", local1.shape) #[128, 256, 32, 50]
      
      local2 = self.c4(out)
      #print("c4 features", local2.shape) #[128, 512, 32, 50]
      out = self.pool(local2)
      #print("local2 features", local2.shape) #[128, 512, 16, 25]

      
      local3 = self.c5(out)
      #print("c5 features", local3.shape) #[128, 512, 16, 25]
      out = self.pool2(local3)
      #print("local3 features", local3.shape) #[128, 512, 16, 12]
      
      out = self.c6(out)
      #print("c6 features", out.shape) #[128, 512, 16, 12]
      out = self.pool(out)
      #print("p6 features", out.shape) #[128, 512, 8, 6]
      
      out = self.c7(out)
      #print("c7 features", out.shape) #[128, 512, 8, 6]
      out = self.pool(out)
      #print("p7 features", out.shape) #[128, 512, 4, 3]
      
      g = self.dense(out) 
      #print("global features", g.shape) #[*, 512, 1, 1]
      
      #pay attention
      if self.att:
          #compatibility score
          #temp = self.project(local1)
          #print(temp.shape)
          cs1, g1 = self.att1(self.project(local1), g) #[*,32,50], [*, 512]
          cs2, g2 = self.att2(local2, g)  #[*,16,25], [*, 512]
          cs3, g3 = self.att3(local3, g)  #[*,16,12], [*, 512]
          g_all = torch.cat((g1, g2, g3), dim=1) # [*, C]
#          print(cs1.shape, g1.shape)
#          print(cs2.shape, g2.shape)
#          print(cs3.shape, g3.shape)
          #classification layer
          out = self.embed(g_all)
      else:
          cs1, cs2, cs3 = None, None, None
          out = self.embed(torch.squeeze(g))
          
      #out = self.pool_final(out)
      #print("emb features, glob features:", out.shape, g.shape) #[*,128], [*,512*3]
     
#      #normalize embeddings to length=1
      #out = self.norm(out, p=2, dim=1) #[*, 128]
      #print("norm emb:", out.shape)
      return out, [cs1, cs2, cs3], [g, g1,g2,g3]

class VisNetAtt3BeforePool1(nn.Module):
  """ Visual subnet """

  def __init__(self, norm):

      super(VisNetAtt3BeforePool1, self).__init__()
      print(self.__class__.__name__)
      self.att = True
      #expected input shape: [batch_size x 3 x 224 x 224]
      out_ch = [64, 128, 128, 256, 512, 512, 128]
      self.c1 = convBlock(in_ch=3, out_ch=out_ch[0], stride=1, norm=norm, num_conv=2)
      self.c2 = convBlock(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, norm=norm, num_conv=2 )
      self.c3 = convBlock(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, norm=norm, num_conv=2)
      self.c4 = convBlock(in_ch=out_ch[2], out_ch=out_ch[3], stride=1, norm=norm, num_conv=2)
      self.c5 = convBlock(in_ch=out_ch[3], out_ch=out_ch[4], stride=1, norm=norm, num_conv=2)
      self.c6 = convBlock(in_ch=out_ch[4], out_ch=out_ch[5], stride=1, norm=norm, num_conv=2)
      
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.pool_final = nn.MaxPool2d(kernel_size=14, stride=14)

      if self.att:
          self.project = projectBlock(out_ch[3], out_ch[6])    #[256, 512] project the global feature dim onto local feat dim
          self.att1 = linearAttBlock(out_ch[6], norm_att=True)
          self.att2 = linearAttBlock(out_ch[6], norm_att=True)
          self.att3 = linearAttBlock(out_ch[6], norm_att=True)         
          glob_dim = out_ch[6] *3
      else:
          glob_dim = out_ch[6]
          
      self.dense = nn.Conv2d(in_channels=out_ch[5], out_channels=out_ch[6], kernel_size=int(224/16), stride=1, padding=0, bias=True)    
      fc_dim = 128
      self.embed = nn.Sequential(
                nn.Linear(glob_dim, fc_dim),
                getNonLinear("relu")       
                )
      self.norm = F.normalize
     
      
  def forward(self, x):
      
      #conv features
      out = self.c1(x)
      #print("c1 features", out.shape) #[*, 64, 224, 224]
      local1 = self.c2(out)
      #print("c2 features", local1.shape) #[*, 128, 224, 224]
      
      out = self.pool(local1)
      #print("p1 features", out.shape) #[*, 128, 112, 112]     
      local2 = self.c3(out)
      #print("c3 features", local2.shape) #[*, 128, 112, 112]

      out = self.pool(local2)
      #print("p2 features", out.shape) #[*, 128, 56, 56]
      local3 = self.c4(out)
      #print("c4 features", local3.shape) #[*, 256, 56, 56]
      
      out = self.pool(local3)
      #print("p3 features", out.shape) #[*, 256, 28, 28]
      out = self.c5(out)
      #print("c5 features", out.shape) #[*, 512, 28, 28]
      out = self.pool(out)
      #print("p4 features", out.shape) #[*, 512, 14, 14]
      out = self.c6(out)
      #print("c6 features", out.shape) #[*, 512, 14, 14]
      g = self.dense(out)
      #g = self.pool_final(out)
      #print("vis features:", g.shape) #[*, 512, 1, 1]
           
      #pay attention
      if self.att:
          #compatibility score
          cs1, g1 = self.att1(local1, g) #[*,112,112], [*, 512]
          cs2, g2 = self.att2(local2, g)   #[*,56,56], [*, 512]
          cs3, g3 = self.att3(self.project(local3), g)   #[*,28,28], [*, 512]
          g_all = torch.cat((g1, g2, g3), dim=1) # [*, 512*3]
          #classification layer
          out = self.embed(g_all)    #[*,128]
      else:
          cs1, cs2, cs3 = None, None, None
          out = self.embed(torch.squeeze(g))
          
      #out = out.view(out.shape[0], -1)  #[*, 512]
      #print("flatten:", out.shape)

      #out = self.embed(out)            #[*, 128]
      #print("embed:", out.shape)
      
      #normalize embeddings to length=1
      #out = self.norm(out, p=2, dim=1) #[*, 128]
      #print("norm:", out.shape)

      return out, [cs1, cs2, cs3], [g, g1, g2, g3]  

