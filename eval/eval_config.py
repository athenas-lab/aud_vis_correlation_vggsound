#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config file for VGGSound self-supervised A-V correlation  model evaluation
"""
import os
import sys
from pathlib import Path


class EvalConfig():

    def __init__(self):
        
       #common config for aud embedding gen and model eval
       self.mgpu =  True
       self.nworkers = 4
       self.norm = "bn"
       self.num_classes = 2 
       self.model = "av_contrast_simclr"
       self.batch_size = 64 
 
       if self.model.startswith("av_contrast"): #contrastive learning models
          self.cont_cfg = {"norm_emb": False, "proj": "linear", "bias": True, "use_cosine": True,  "pos_denom":True}
       
    
       #self.sfx_feat_list = "lists/gt_aud.lst" #"lists/afeats.lst"
       self.win_len=100   #audio frame length used by trained model
       self.aud_frames=100 #audio length used for embedding

       #self.aud_list = self.vis_list = "../lists/vggsound_test.txt"
       self.aud_list = self.vis_list = "../lists/vggsound_val.txt"      
       self.vis_feat = {"type": "raw", "fps": 1}
       #config for model evaluation
       #ways to evaluate the trained A-V correlation model:
       #["gen_aud_emb",  "img_sim", "aud_sim", "sound_rec"] 
       #self.demo = "img_sim"  
       self.demo = "sound_rec"  
       self.topk =  10
       
       self.img_size = 224
       self.eval_preds = True #True: evaluate predictions if you have ground truth. False: Dont evaluate preds, just generate them.
       info = self.getPaths()
       self.saved_weights =  info["saved_wts"]                        
       self.out_file =  info["out_file"]     
       p = Path(info["out_file"]).parent
       if os.path.exists(p) == False:
           os.makedirs(p)   
    
   
    def getPaths(self):

        
        saved_weights = "../vggsound_eval_root/saved_models/av_contrast_simclr/files_tag_noisy_aud_fb_vad/tr_vggsound_train_val_vggsound_val/raw_1fps/loss_ntxent/lr_0.01_cosine_linearproj_posden_bias_temp0.5_bs32/checkpoints/best_model_wts.pth"
        of = "av_contrast_simclr/sound_rec.json"
        
        path_info = {} 
        path_info["saved_wts"] =  saved_weights
        #path_info["aud_emb"]   =  os.path.join(sdp["sfx_emb_from_av_model_root"], aud_emb)
        out_dir = os.path.join("vgg_results/%s"%self.demo)
        path_info["out_file"]  = os.path.join(out_dir, of)
    
    
        return path_info
    
  
