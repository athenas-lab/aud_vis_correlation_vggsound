#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Main configuration file for audio-visual correlation 
"""

import os
import json

import utils

class Config():
    """ set hyperparams and other configurations """

    def __init__(self, mode, platform, aud_tag, vis_feat, data_split, mdl, demo):
        
        #Formulate as a binary classification task.
        #0: a-v are not related, 1: a-v are related
        self.num_classes = 2 #2for a/v correlation; 169 for multi-label obj classification 
        self.mode = mode #["train", "test"]
        self.demo = demo
        #False= do not use pretrained model
        self.load_pretrained_model = self.mode != "train" 
        self.model = mdl
        self.platform = platform
        self.aud_tag = aud_tag     
        self.mean_norm = True #normalize the audio data using the training mean/std
        self.norm = "bn"  #batch norm or None
        self.data_size = {}
        self.vfeat = vis_feat
        self.topk = None #20
        self.mgpu = True #use multi-gpu
        
        self.att = False
        if self.att:
           self.att_cfg = {"norm_attn": True, "num_att": 3, "disp_att":True}
        self.cont_cfg = {"norm_emb": False, "proj": "linear", "bias": True, "use_cosine": True,  "pos_denom":True}
        self.tagConfig()
        
        #used for RNN models
        self.seq_len =  self.getVisRate()  
        if self.model.startswith("av_contrast_simclr"): 
           self.margin =  0.5  #temperature
           self.loss = "loss_ntxent"    
        else:

           self.margin = 0.1
           self.loss = "loss_bce"  #binary x-entropy
           
    def tagConfig(self):
        """ Configure tag attributes """ 
        
        self.selfSuperTagConfig()

        
        return      
     
    def selfSuperTagConfig(self):
        """ File-level/subcat tag config for self-supervised video tagging models"""
         
        self.tag_map = None
        #indicates how the subcat/file-level tags are generated
        tag_src = self.getTagSrc()
        self.tag_map = "noisy_aud"
           
        return 
    
   
    def pathConfig(self, data_split, misc = ""):
        misc = misc + "_%s"%("cosine" if self.cont_cfg["use_cosine"] else "dot")
        misc = misc + "_%s"%self.cont_cfg["proj"] + "proj"
        misc = misc + "_%s"%("posden" if self.cont_cfg["pos_denom"] else "negden")
        misc = misc + "%s"%("_normemb" if self.cont_cfg["norm_emb"] else "")
        misc = misc + "%s"%("_bias" if self.cont_cfg["bias"] else "")
        misc = misc + "_temp%s"%(str(self.margin))  + "_bs%s"%(str(self.batch_size))
        eval_root = "./vggsound_eval_root" #for vggsound dataset training
        if self.mode != "test": 
            data_str = utils.getDataStr(data_split)
            lr_str= "lr_{}".format(self.learn_rate)    
        
            vis_feat = utils.getVfeatStr(self.vfeat)
            tag_str = utils.getTagStr(self.aud_tag)
            suffix =  os.path.join(self.model, tag_str,  data_str,  vis_feat, self.loss, lr_str+misc)  

        self.model_root = os.path.join(eval_root, "saved_models", suffix)
        self.result_dir = os.path.join(eval_root, "results", suffix)
        self.model_log_dir= os.path.join(self.model_root, "logs")
        #print("model path = %s, result dir = %s, log_dir = %s"%(self.model_path, self.result_dir, self.log_dir))
        
        self.model_checkpt_dir = os.path.join(self.model_root,"checkpoints")
       
        #path to saved weights for evaluation
        if self.platform == "pt":
            self.saved_weights = os.path.join(self.model_checkpt_dir, "best_model_wts.pth")
        #path to previous checkpoint for resuming training    
        self.restore_model_ckpt = None 
        
        #suffix for logging
        if self.mode=="test" and self.demo is not None:
            suf = "_demo_%s"%str(self.demo)
        else:
            suf = ""
        self.log_file = os.path.join(self.result_dir, "%s_logs%s.txt"%(self.mode, suf))
        self.result_file = os.path.join(self.result_dir, "%s_tags%s.txt"%(self.mode, suf))   
        self.writer_out = os.path.join(self.result_dir, "%s_summary%s"%(self.mode, suf))
        if self.mode != "test":
            if os.path.exists(self.result_dir) == False:
                os.makedirs(self.result_dir)
            if os.path.exists(self.model_checkpt_dir) == False:
                os.makedirs(self.model_checkpt_dir)
          
        return
    
    def modelConfig(self):
        return
    
    
    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        return

    def getVisType(self):
        if self.vfeat:
          return self.vfeat["type"]
        else:
          return None
        
    def getVisRate(self):
        if self.vfeat:
           return self.vfeat["fps"]
        else: 
            return None
        
    def getVis2TagMap(self):
        return self.tag_map
    
    def getTagSrc(self):
        if self.aud_tag:
           return self.aud_tag["src"]
        else:
            return None
        
    def getTagLevel(self):
        if self.aud_tag:
           return self.aud_tag["level"]
        else:
            return None
    
    def getTagFeat(self):
        if self.aud_tag:
           return self.aud_tag["feat"]
        else:
            return None
    
class ConfigAudImg(Config):
    """ Config for audio-image training """
    def __init__(self, mode, platform, aud_tag, vis_feat, data_split, mdl, demo=False):
       #video frame rate(1 or 30), if feat_type = "video". For image this is fixed at 1fps.
       Config.__init__(self, mode, platform, aud_tag, vis_feat, data_split, mdl, demo) 
       self.feat_ext = False #True: enable transfer learning
       self.use_random_flip = True #for video/image augmentation
       self.modelConfig()
       self.pathConfig(data_split)
       
    def modelConfig(self):
        """Loads model configuration  from json file"""
        
    
        #self.bn_momentum = 0.9
        self.num_epochs = 1000
        #self.learn_rate = 1e-06 #for multi-label
        self.learn_rate = 0.01
        #batch_size is the total batch size. so when using 2 gpus, the bs for each gpu is bs/2.
        #self.batch_size = 96 #obj detect
        #self.batch_size = 16 # for att, 4 for 1 gpu 
        self.batch_size = 32
        #self.batch_size = 128  #normal
        
        self.dropout_p = 0.5  # what fraction of neurons to keep active (1-dropout)
        self.reg_lambda = 0.0  # regularization factor
        self.load_to_mem= False # Preload data to memory (set to False in case of large data) 
        self.num_checkpoints = 1 #how many check points to keep
        self.eval_every = 1  #run validation at this periodicity during training
        self.early_stop = 20 #how many epochs to wait for improvement in validation performance       
        self.save_summary_period = 1 #periodicity for saving model summary
        
        #data related config
        self.nworkers = 4 #8   #for pytorch dataloader
        self.img_size = 224  #image size for non-Inception model input
        self.aud_frames = 100 #window length for training (100 frames=1sec)
           
        return
    
    
       
