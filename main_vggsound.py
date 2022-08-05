import logging
import os
import sys
import time

import numpy as np
import random


 

#import the appropriate models
sys.path.append("data")

import utils
import config

""" Main file for self-supervised sound recommendations using VGGSound dataset"""

def printConfig(conf):
    
    conf = conf.__dict__
    mdl_root = conf["model_root"]
    
   
    logging.info("===========Starting at time: {}======".format(time.ctime()))
    logging.info("Logging file = %s"%(conf["log_file"]))
    logging.info("Result dir = %s"%(conf["result_dir"]))
    for k, v in conf.items():
        logging.info("{} = {}".format(k, v))

    print("===========Starting at time: {}======".format(time.ctime()))
    print("Model dir = %s"%(mdl_root))
    print("Logging file = %s"%(conf["log_file"]))
    print("Result dir = %s"%(conf["result_dir"]))
#    print("Number of tag classes = %d"%(conf["num_classes"]))
    return
    
def testLoader(dl):
    
      #for k  in dl.keys():
      tr = dl["train"]

      #print("=====",k, "======")
      for sample in tr:
        for k in sample.keys():
            #print(k, sample[k])
            if k == "aud":
               print(sample[k].shape)
     
        break     
      return
    
def soundRecSelfSuper(conf, dsets, mode):
    """ 
        Train self-supervised models to learn A,V correlation to recommend sounds for video. 
    """     
   
    
    utils.setLogger(conf.log_file)
    printConfig(conf)
    
    #get the dataloader dictionary for the data splits
    dl_dict = {}
    if conf.mode == "test":
       modes = ["test"]
    else:   
       modes =  ["train", "val", "test"]
    
    if conf.model.startswith("av_contrast"): #contrastive learning models    
       #import data_loader as dl  
       import data_loader_contrast as dl
       
    av_noisy_loader = None
    for mode in modes:
      
         if av_noisy_loader is None:  
            av_noisy_loader = dl.VideoNoisyAudLoader(conf)              
         dl_dict[mode] = av_noisy_loader.getDataLoader(dsets, mode)

    #testLoader(dl_dict) 
    #exit()  
               
    #initialize random seed
    random.seed(144)
   
    if conf.model.startswith("av_contrast"): #contrastive learning models
       import train_vgg_contrast as train_eval
    elif conf.model == "av_simclr_cor":
        import train_vgg_simcor as train_eval
       
    if conf.mode == "train":
       train_eval.trainModel(conf, dl_dict)
    else:
       train_eval.evalSavedModel(conf,  dl_dict["test"])
    
    print("Logging file = %s, \n Tag_file=%s"%(conf.log_file, conf.result_file))
    return



def main():

    mode= "train"  #["train", "test", "demo"]
    platform = "pt" #["tf", "pt", "keras"]
    
    #audio tag description
    aud_tag = {
            "level": "files",  
            "src":  "noisy_aud", # self-supervision from video 
            #[audio tag features in case of "file" level: filterbank with VAD("fb_vad")]
            "feat": "fb_vad"
          }
    
    #visual feature description
    vis_feat = {
            #[type of visual features
            "type": "raw", 
            #video frame rate. [1(image), 30]
            "fps": 1
          }
    
    #dataset identifiers   
    #train directly using noisy audio
    dsets= {
                "train": ["vggsound_train"],    
                "val"  : ["vggsound_val"],
                "test" : ["vggsound_test"] 
               }

    #contrastive learning model
    model_name = "av_contrast_simclr"

    #set hyperparams and other configurations
    if  vis_feat["fps"] == 1:      
       conf = config.ConfigAudImg(mode, platform, aud_tag, vis_feat, dsets, model_name)
  
    #  Train self-supervised models to recommend file-level/subcat tags for video. 
    soundRecSelfSuper(conf, dsets, mode) 

    return


if __name__ == "__main__":

   main()

 
