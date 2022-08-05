import torch
import torch.nn as nn
import torch.nn.functional as F

import init_weights as iw
from backbone import *

""" Self-supervised AV correlation models using SimCLR contrastive learning with different backbones"""


    
class AVSimCLR(nn.Module):
  """ Implements the contrastive learning model described in the SimCLR paper """  
  
  def __init__(self, cfg, norm, init_wts="xav_uni"):
         
      super(AVSimCLR, self).__init__()
      print("AVSimCLR")

      #flag to normalize the representations before projection
      self.norm_emb = cfg.cont_cfg["norm_emb"]    
      #flag to include bias in the final MLP layer of the projection net
      self.bias = cfg.cont_cfg["bias"]    
      
      #visual representation
      self.vfeatures = VisNetAtt3BeforePool1(norm=norm) 
      #audio representation
      self.afeatures = AudNetAtt3BeforePool2(norm=norm) 
      #print(self.vfeatures)    
      #siamese MLP layer to project  A and V embeddings    
      if cfg.cont_cfg["proj"] == "nonlin":          
          self.proj = nn.Sequential(
                    nn.Linear(128, 128, bias=True),  
                    getNonLinear("relu"),
                    nn.Linear(128, 128, bias=self.bias)
                    )
      elif cfg.cont_cfg["proj"] == "linear":  
          self.proj = nn.Linear(128, 128, bias=self.bias)
          #use this for posden linproj_0.5_bs16 (initial model)       
#          self.proj = nn.Sequential(
#             nn.Linear(128, 128, bias=self.bias)
#               ) 
      
      
      self.norm = F.normalize   
      iw.initWeights(self, init_wts)
     
      
  def forward(self, x_v, x_a):
      #print("==========")
      v_emb, v_c, v_g = self.vfeatures(x_v)
      a_emb, a_c, a_g = self.afeatures(x_a)
      if self.norm_emb:
          v_emb = self.norm(v_emb, p=2, dim=1)
          #print("v_emb:", v_emb.shape) 
          a_emb = self.norm(a_emb, p=2, dim=1)
          #print("a_emb:", a_emb.shape)
  
      #v_emb = v_emb.squeeze()
      #a_emb = a_emb.squeeze()

      #normalized projections for A and V 
      v_proj = self.proj(v_emb)
      v_proj = self.norm(v_proj, p=2, dim=1) #[*, 128]
      a_proj = self.proj(a_emb)
      a_proj = self.norm(a_proj, p=2, dim=1) #[*, 128]
       
      return v_emb, v_proj, a_emb, a_proj


  def getImgEmb(self, x_v):
      """Get visual embeddings for a given image"""
     
      v_out, v_c, v_g = self.vfeatures(x_v)
      #if original model did not emit normalized embeddings, then normalize
      if not self.norm_emb:
         v_out = self.norm(v_out, p=2, dim=1) #[*,128]
      #v_emb = torch.squeeze(v_g[0])
      #print(v_emb.shape)
      
      return v_out
  
  def getAudEmb(self, x_a):
      """Get audio embeddings for a given audio sample"""
      
      a_out, a_c, a_g = self.afeatures(x_a)
      #if original model did not emit normalized embeddings, then normalize
      if not self.norm_emb:
        a_out = self.norm(a_out, p=2, dim=1) #[*,128]
      #a_emb = torch.squeeze(a_g[0])
      #print(a_emb.shape)
      return a_out

 
class AVSimclrCor(nn.Module):  
   """ Learning correlation after the contrastive model has been trained """
   def __init__(self, cfg, norm, init_wts="xav_uni"):
         
      super(AVSimclrCor, self).__init__()
      print("AVSimclrCor")  
      #path to the trained contrastive model 
      pre_path = "vggsound/raw_1fps/loss_ntxent/lr_0.01_noaug_cosine_linearproj_posden_bias_temp0.5_bs16/checkpoints/best_model_wts.pth"

      self.num_class = cfg.num_classes
      self.av_model = AVSimCLR(cfg, norm)
#      self.vfeatures = self.av_model.vfeatures
#      self.afeatures = AVSimCLR(cfg, norm).afeatures
      #0/1 binary classifier.  1 if audio is related to the visual, else 0.
      self.av_classifier = nn.Linear(1, self.num_class, bias=True)          
      self.load_state_dict(torch.load(pre_path), strict=False)
      
   def forward(self, x_v, x_a):   
      #visual representation
      v_out = self.av_model.getImgEmb(x_v) 
      #audio representation
      a_out = self.av_model.getAudEmb(x_a)
      #euclidean distance between embeddings
      av_dist = F.pairwise_distance(v_out, a_out)
      #print("av_dist:", av_dist.shape)
      
      out = av_dist.view(av_dist.shape[0], 1)
      #print("extend:", out.shape)

      out = self.av_classifier(out)
      #print("av_classifier:", out.shape)
      return out, av_dist, v_out, a_out

  
   def getEmbCor(self, x_1, x_2):
      """Get correlation between input embeddings """
      #print(x_1.shape, x_2.shape) 
      #euclidean distance between embeddings
      dist = F.pairwise_distance(x_1, x_2) #sqrt( (x_1-x_2).pow(2).sum())
      #dist = F.mse_loss(x_1, x_2, reduction='none').mean(1)
      #print("dist:", dist.shape)
      out = dist.view(dist.shape[0], 1)
      #print("extend:", out.shape)
      out = self.av_classifier(out)
      out = F.softmax(out, dim=1)
      
      return dist, out
