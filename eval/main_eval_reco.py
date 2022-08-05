import logging
import os
import sys
import time
import json
from collections import namedtuple

import numpy as np
import random


""" Main file for evaluating audio-video correlation model 
    Model can be used for:
        image similarity 
        audio similarity 
        audio recommendations for video
        visual recommendations for audio
"""


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

    for sample in dl:
        if "v" in sample.keys():
           v = sample["v"]
           f = sample["v_fn"]
           print("vis_feat", v.shape, f, sample["label"])
        if "a" in sample.keys():
           a = sample["a"]
           f = sample["a_fn"]
           print("aud_feat", a.shape, f, sample["label"])

        break
    return

def getAudList(list_file):
       """ Get test data list with ground-truth """
       
       frames = []
       labels = []
       
       with open (list_file, "r") as fin:
           for line in fin:
              f, lab = line.strip().split("|")
              #Extract number of video frames from file names which have format: name_nframes. 
             
              frames.append(f) #select first frame
              labels.append(lab)
       assert(len(set(frames)) == len(frames)) #ensure the frame list has no duplicates
       print("aud eval data:number of samples=%d"%(len(frames)))          
       return frames, labels   
     
def getVisList(list_file):
       """ Get test data list with ground-truth """
       
       frames = []
       labels = []
       #provide the path to the VGGSound video frames
       frame_root = "data/vggsound/frames_1fps"
       with open (list_file, "r") as fin:
           for line in fin:
              f, lab = line.strip().split("|")
              #Extract number of video frames from file names which have format: name_nframes. 
              if os.path.exists(os.path.join(frame_root, f) + "/image_005.jpg"):
                 frames.append(f+"/image_005.jpg") #select first frame
              else:
                  frames.append(f+"/image_001.jpg") #select first frame
              labels.append(lab)
#              video_frames = sorted(glob.glob(os.path.join(frame_root, f) + "/*.jpg"))
#              nframes = len(video_frames)
#              if nframes > 1: #if video has more frames, select a few more 
#                 nf = int(nframes/2) + 1
#                 frames.append(f+"/image_{:03}.jpg".format(nf))
#                 labels.append(lab)
       assert(len(set(frames)) == len(frames)) #ensure the frame list has no duplicates
       print("vis eval data:number of samples=%d"%(len(frames)))          
       return frames, labels   
   
   
def getVisLoader(config, data):
    
    import vis_loader as el
     
    eval_dl = el.getImgLoader(config, data) 
    return eval_dl

def exeDemo(config):
    
    import eval_self_super_av as ess
    
    demo = config.demo
    
    if demo == "img_sim":  
       data = getVisList(config.vis_list)    
       eval_dl = getVisLoader(config, data)      
       ess.evalImgSim(config, eval_dl)
    elif demo == "img_emb":  
       data = getVisList(config.vis_list) 
       eval_dl = getVisLoader(config, data) 
       #testData(eval_dl) 
       ess.genImgEmb(config, eval_dl)
    elif demo == "aud_emb": 
       #generate the audio embeddings    
       data = getAudList(config.aud_list) 
       ess.genAudEmb(config, data) 
    elif  demo == "sound_rec":  
       #generate audio and visual embeddings and 
       #recommend appropriate audio for video using pretrained model
       vis_data = getVisList(config.vis_list)         
       eval_dl = getVisLoader(config, vis_data) 
       #testLoader(eval_dl)    
       aud_data = getAudList(config.aud_list)
       ess.genAudioRec(config, eval_dl, vis_data, aud_data)
    print ("Result file =%s"%config.out_file)    
    return





if __name__ == "__main__":

   import eval_config as ec
   
   conf = ec.EvalConfig() 
   print(conf.saved_weights)
   start = time.time()
   print(start)
   exeDemo(conf)
   print("elapsed time", time.time() - start)

 
