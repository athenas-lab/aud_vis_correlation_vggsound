#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" VGGSound Audio-Visual  dataset processing for  training audio-visual correlation models"""

import os
import sys
import logging
import re
import json
import glob

from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset

#sys.path.append("../../..")

#root folder for vgg visual and audio data
aud_root = "data/vggsound/audio_feats/fb_48k"
frame_root = "data/vggsound/frames_1fps"

def getFrames(f):
    #return the frames in the video 
    frames = []
    #Extract number of video frames from file names which have format: name_nframes. 
    #frames.append(f+"/image_001.jpg") #select first frame
    video_frames = sorted(glob.glob(os.path.join(frame_root, f) + "/*.jpg"))
    nframes = len(video_frames)
    #if nframes > 1: #if video has more frames, select a few more 
    #   nf = int(nframes/2) + 1
    for nf in range(nframes):
          frames.append(f+"/image_{:03}.jpg".format(nf+1))
    assert(len(set(frames)) == len(frames)) #ensure the frame list has no duplicates
    #print("number of samples=%d"%(len(frames)))
    return frames

def getCorAud(img_path):
     #get the audio file  and starting frame number
     afile = img_path.split("/")
     fn = afile[-1]  #get visual frame number
     fn, _ = os.path.splitext(fn)  #remove image extension
     #audio frame starts from 0, vframe starts from 1. So aframe = vframe -1
     fn = int(fn.rsplit("_") [1])-1  
     #fn = 0
     afile = "/".join(afile[:-1]) + "_" + str(fn)    
     return afile
 
def getDataList(v2a_map):
    """ Get the data list from the video to pos/neg aud mapping"""
    
    #video to audio mapping
    with open(v2a_map, "r")  as fp:
       v2a = json.load(fp)
    data_list = []   
    for k, v in v2a.items():
        sample = {"vis": k, "label": v["label"]}
        data_list.append(sample)

    return data_list


class VGGSound(Dataset):
    
    """ Iterates over the game data frame file list and returns the data necessary for 
        training audio-visual correlation models.
    """
    def __init__(self, conf, data_file, vis_transform):
       
       self.conf = conf
       self.vis_transform = vis_transform
       self.aud_transform = self.getAudTransform(self.conf.mean_norm)

       
       #list of frames and corresponding audio label (single label for pairwise sim)
       self.data_list = getDataList(data_file)
       
       # Map visual files to labels consisting of sfx file indices 
       #self.filterFrames(frames, labels)       
       print("loaded labels2vid:{}".format(len(self.data_list))) 
       logging.info("loaded vis2aud_idx:{}".format(len(self.data_list)))  

       np.random.seed(144)
       random.seed(144)
       
    def filterFrames(self):
        """ Get a subset of the data"""

       
        #limit the training data
        if (not self.conf.demo) and len(self.frames) > 150000:      
          #select a smaller subset  
          random.seed(144)
          #print(len(self.frames))
          #rind = random.sample(range(len(self.frames)), 1500)  
          rind = random.sample(range(len(self.frames)), 150000)  
          self.frames = [self.frames[i] for i in rind]
          self.labels = [self.labels[i] for i in rind]

        print("Reduced num frames = {}. num labels={}".format(len(self.frames), len(self.labels)))

        return   

       
    def __len__(self):
       """ return length of the dataset """
      
       return len(self.data_list)
   

    def globalContrastNorm(self, img):
        """ global contrast pixel normalization """
        
        eps = 1e-09
        lmda = 5
        s = 1
        img_res = img.resize((224, 224))
        x = np.array(img_res)
        x_avg = np.mean(x)
    
        x_new = x - x_avg
        contrast = np.sqrt(lmda + np.mean(x_new**2))
        new_img = s*x_new/ max(contrast, eps)
        new_img = np.transpose(new_img, (0,2,1))
        new_img = np.transpose(new_img, (1,0,2))
        new_img = torch.from_numpy(new_img).float()
        return new_img
    
    def getVisFrame(self, img_path):
       #return video frame given the path 
        
       
       ip = os.path.join(frame_root, img_path)
       x = Image.open(ip)
       #print(img_path, "original shape=", x.shape)
       #x = self.globalContrastNorm(x)
       
       #apply data transform if required
       if self.vis_transform is not None:
       #   #print(x.size)
          x = self.vis_transform(x)
       return x   
     
    def getAudTransform(self, mean_norm):
        """ Get audio transform parameters, if required """
        
        params = {} 
        params["aud_frames"] = self.conf.aud_frames
        
        #mean normalization  for noisy audio
        if mean_norm:
            #normalize, if filterbank audio features are used as labels
            afeat = self.conf.getTagFeat()
            if afeat.startswith("fb_vad"):
               #load the precomputed mean and variance for the training features 
               mv = (np.load("lists/vggsound_train_mean_var.npy"))
        else:
            mv = None
        if mv is not None:
                  params["mean"] = mv[0]
                  eps = 1e-8
                  params["std"] = np.sqrt(mv[1]) + eps
   
        return params

    
    def transformAud(self, afeat, params):
        """ 
            If the number of feature frames is < 100, we pad by replicating the 
            features so that the number of frames = 100. This is to ensure 
            that we can use  1 sec of audio feature for the clean label 
            (1 sec = ~94 frames).
            If the audio is longer than 1 sec, a random 1 sec sample is extracted.
            Audio feature is then normalized using training data's mean and var.
            Features are transposed and expanded to get a channel first format.
        """
        
        
        mod_feat = afeat
        #print("afeat shape", afeat.shape)
        if params is not None:
          nframes = params["aud_frames"] #default= 100, corresponds to 1 sec audio
          #restrict window length to 1 sec for training. No restriction during evaluation or embedding generation.
          if nframes is not None: 
            #pad by wrapping the features.   #smallest number of audio frames = 47
            if afeat.shape[0] < nframes:
              mod_feat = np.pad(mod_feat,((0, nframes-afeat.shape[0]), (0, 0)), mode='wrap')
              #print("padding")
              #   mod_feat = np.concatenate((mod_feat, afeat), axis=0)
            elif afeat.shape[0] > nframes: #pick a random 1sec audio from the clean sfx if it is longer than 1 sec.
              #start_idx = np.random.choice(range(0, (afeat.shape[0]-nframes+1)))
              #mod_feat = mod_feat[start_idx:start_idx+nframes,:]  
              mod_feat = mod_feat[:nframes,:]  #first nframes to make the model predictable
          if "mean" in params.keys():    
             #normalize the audio sample. #standard normalization
             mod_feat = (mod_feat - params["mean"])/params["std"] 
             #print(mod_feat)
        mod_feat = mod_feat.astype(float)     
        #transpose to (feature dim X num frames) format
        mod_feat = np.transpose(mod_feat)
        #expand dim to get (channel X feature dim X num frames) format. channel=1.
        #This will allow the audio data to be input to a 2D conv net.
        mod_feat =  np.expand_dims(mod_feat, axis=0)  
        return mod_feat                 
         
    
    def getAudioLabel(self, afile, frame_num=0):
       """ Get the audio label features """
       
       
       afeat = self.conf.getTagFeat()
       if afeat.startswith("fb_vad"):
          y = np.load(afile)
          #print(afile, y.shape)
          #entire video in vggsound data has the same audio label. So instead of 
          #just extracting 1 sec audio corresponding to the video frame, we average the sound features 
          #over the entire video duration (frame_num = video duration)
          start = frame_num*100
          if start > y.shape[0]:
              start = 0 * 100 #get the first 1 sec if audio is shorter then the video length
          y = y[start:] #start+frame_num*100]
          #print(afile, frame_num, y.shape)
          #avg_1sec = np.zeros((100, y.shape[1]))
          #for i in range(frame_num):
          #    avg_1sec = np.sum(avg_1sec, y[i*100: (i+1)*100])
          #audio features with fixed frame length, normalized, transposed, and expanded to 3D.  
          y = self.transformAud(y, self.aud_transform)
         
       return y

    def getNegativeLabel(self, vid, lab):
      """ Get a negative label (audio unrelated to video)"""

      #get all the labels that dont match the video label
      not_labs = list(set(list(self.labels2vid.keys())) - set([lab]))
      #pick a random label from the complement      
      neg = np.random.choice(not_labs)
      #get the videos having this negative label
      not_labs = self.labels2vid[neg]
      #pick a random video from the complement list      
      neg = np.random.choice(not_labs)
      #extract name and frame number
      #print("neg", neg)
      #neg, fn = neg.rsplit("_", 1)
      fn = 0 #get 1 sec audio from start (can also be a random 1sec audio)
      #get the audio path for the selected video
      neg_file = os.path.join(aud_root, neg)+".npy" 
   
      return neg_file, fn
   
    
      
    def __getitem__(self, n):
       """ Return a (single visual frame, matching audio for positive label, visual frame filename, label) tuple 
       """  
       element = self.data_list[n]
       video = element["vis"]
       vframes = glob.glob(os.path.join(frame_root, video) + "/*.jpg")
       #pick a random video frame for the video       
       img_path = np.random.choice(vframes) 
       x = self.getVisFrame(img_path)
       #print(element, img_path)
       
       #get corresponding audio frame
       frame = img_path.split("/")[-1]
       afn = getCorAud(video+ "/" + frame)
       afn = afn.rsplit("_")
       #print("afn split", afn) 
       afile =  os.path.join(aud_root,"_".join(afn[:-1])) +".npy"      
       fn=int(afn[-1])
       #print("afile", afile, "fn", fn) 
       assert (os.path.exists(afile))
       y = self.getAudioLabel(afile, fn) 
       afile = os.path.splitext(afile)[0] + "_%s.npy"%(str(fn)) 
       #print(afile)
        
       sample = {"vis": x, "aud":y, "v_fn": img_path, "a_fn": afile, "label": element["label"]}  
          
       #print(sample)
              
       return sample
   
    
    
