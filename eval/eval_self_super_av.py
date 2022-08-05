#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation methods for sound recommendation and A-V correlation.
"""

import time
import os
import copy
import sys
import logging
import json   
import glob 
  
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as ed

import torch

sys.path.append("../models")
from init_model import getModel 
import gen_aud_embs as gae
    
class EvalMMDEmbed():
  """ Loads the specified trained model and provides methods to generate audio/visual embeddings """
  
  def __init__(self, config):
     
        self.config = config
        self.getTrainedModel()
        
        #load the mapping from labels to video files for evaluation
        with open("../lists/labels2vid.json") as fin:
           self.labs2vid = json.load(fin)
    
  def getTrainedModel(self):
        #load the trained model 
        model = getModel(self.config)      
        #if use_cuda:
           #  gpu_dtype = torch.cuda.FloatTensor
           #  self.model = self.model.type(gpu_dtype)
        self.use_cuda = torch.cuda.is_available() # check if GPU exists
        self.device = torch.device("cuda" if self.use_cuda else "cpu") # use GPU or CPU
        num_gpus = torch.cuda.device_count()
        gpu_list = None
        if self.config.mgpu and num_gpus > 0: #use multi-gpus
             print("Using %d gpus"%num_gpus)
             gpu_list = list(range(num_gpus))
             model = torch.nn.DataParallel(model, device_ids=gpu_list)
      
        self.gpu_list = gpu_list
        model.to(self.device) 
        self.model = model
        if self.model:
           model_wts = self.config.saved_weights
           logging.info("Loading model weights from %s"%model_wts)
           print("Loading model weights from %s"%model_wts)
         
        #first deserialize the state dictionary by calling torch.load()  
        if self.use_cuda: #GPU  
           new_sd = torch.load(model_wts)
        else: #CPU
            sd = torch.load(model_wts, map_location='cpu') 
            new_sd = {}
            #trained model uses DataParallel. 
            #so remove "module" from the model parameters.
            for k, v in sd.items():
                    import re
                    #remove "module" in state dictionary keys
                    k = re.sub("module\.", "", k)
                    #print(k, v)
                    new_sd[k] = v
        self.model.load_state_dict(new_sd, strict=True)
        #set the model to evaluate mode
        self.model.eval()
        #count the number of trainable params
        #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print(num_params)
        #exit()
           
           


  def genImgEmb(self, data_loader):
    
        """ Get image embeddings from pretrained AV correspondence model """
        
        logging.info("getImgEmb")
        model = self.model
      
        all_fn = [] #visual file names
        all_emb = [] #visual embeddings
        #vid = np.random.normal(size=(4, 3, 224, 224))
        #aud = np.random.normal(size=(64, 1, 64, 100))
        #vid = torch.autograd.Variable(torch.from_numpy(vid))
        #aud_var = torch.autograd.Variable(torch.from_numpy(aud)).to(self.device, dtype=torch.float)
        with torch.no_grad():
             for  sample in data_loader:
                v_dev = sample["v"] #visual feats
                #print(v_dev)
                v_dev = v_dev.to(self.device, dtype=torch.float)
                
                if self.use_cuda:
                  emb = model.module.getImgEmb(v_dev)
                else:
                  emb = model.getImgEmb(v_dev)  
                #aud = np.random.normal(size=(v_dev.size(0), 1, 64, 100))
                #aud_var = torch.autograd.Variable(torch.from_numpy(aud)).to(self.device, dtype=torch.float)
                #logits, dist, emb, aemb = model(v_dev, aud_var)
                #print(emb, sample["v_fn"])
                #dist, cor = model.module.getEmbCor(emb, emb)
                #print("emb distance", dist.data.cpu().numpy())
                #print("emb corr", cor.data.cpu().numpy(), torch.argmax(cor, 1))
                all_emb.extend(emb)
                all_fn.extend(sample["v_fn"]) #visual file name
                
                
        logging.info("getImageEmb:Test size: %d"%(len(data_loader.dataset)))        
        all_emb = torch.stack(all_emb, dim=0).cpu().data.squeeze().numpy()    
        #all_emb = np.random.normal(size=(676, 128))
        print("all embeddings shape", all_emb.shape)
        return all_emb, all_fn

  def genAudEmb(self, data_loader, save=False):
    
        """ Get audio embeddings from pretrained AV correspondence model """
        
        logging.info("getAudEmb")
            
        model = self.model
        
        all_fn = [] #audio file names
        all_emb = [] #visual embeddings
        
        with torch.no_grad():
             for  sample in data_loader:
                a_dev = sample["a"] #audio  feat
                a_fn = sample["a_fn"] #audio file names  
                a_dev = a_dev.to(self.device, dtype=torch.float)
                if self.use_cuda:
                   emb = model.module.getAudEmb(a_dev)
                else:
                    emb = model.getAudEmb(a_dev)
                print(emb.shape, sample["a_fn"])
                if not save:
                    all_emb.extend(emb)            
                    all_fn.extend(a_fn) #visual file name
                else: 
                    outEmb(emb, a_fn)
                break
               
        logging.info("getAudEmb:Test size: %d"%(len(data_loader.dataset))) 
        if not save:       
            all_emb = torch.stack(all_emb, dim=0).cpu().data.squeeze().numpy()    
            print(all_emb.shape)
            return all_emb, all_fn
        return 
    
  def compareEmb(self, e1, e2):

      e1 = torch.from_numpy(e1).to(self.device, dtype=torch.float)
      e2 = torch.from_numpy(e2).to(self.device, dtype=torch.float)
      model = self.model
      with torch.no_grad():   
         if self.use_cuda:                    
            dist, cor = model.module.getEmbCor(e1, e2)
         else:
            dist, cor = model.getEmbCor(e1, e2) 
         cor = torch.argmax(cor, 1)
         print("dist, cor", dist, cor)
      return     
  
  def compareAVEmb(self, e1, e2):
     
      ae = torch.from_numpy(e2).to(self.device, dtype=torch.float)
      model = self.model
      va_cor = {}
      with torch.no_grad():
        for i in range(len(e1)):
           #compare visual embedding with all of the audio embeddings to get correlated audio 
           ve = np.asarray([e1[i]]*(e2.shape[0]))
           ve = torch.from_numpy(ve).to(self.device, dtype=torch.float)
           if self.use_cuda:     
              dist, cor = model.module.getEmbCor(ve, ae)
           else:
              dist, cor = model.getEmbCor(ve, ae)
           #print("v, dist, cor", i, dist, cor)
           #cor = torch.argmax(cor, 1)
           #print("cor argmax",  cor)
           
           dist = list(dist.cpu().data.squeeze().numpy()) 
           cor = cor.cpu().data.squeeze().numpy() 
           print("dist", dist)
           print("cor", cor)
           va_cor[i] = {"cor_ind":[], "cor_dist": []}
           va_cor[i]["cor"] = list(np.argmax(cor, 1))
           va_cor[i]["dist"] = [float(d) for d in dist]
                
      #print(va_cor)
      return  va_cor   

def getKNN(src_emb, dest_emb, topk):
    """ For a given list of source embeddings, 
        find the k-nearest nbrs from the destination embeddings. 
    """
    
    top_ind = []
    top_dist = []
    
    #get the euclidean distance between the source and dest embeddings
    dist = np.squeeze(ed(src_emb, dest_emb))
    if dist.ndim == 1:
            dist = dist.reshape(1,-1)
   
    #get the top-k nearest nbr indices for each source     
    top_ind = np.argsort(dist, axis=1)[:,:topk]
    #print("dist, dist shape", dist, dist.shape)
    top_dist = np.sort(dist, axis=1)[:,:topk]
  
#    print("KNN: vis emb", src_emb)
#    print("KNN: aud_emb", dest_emb)
    print("topk ind", top_ind)
    print("topk dist",top_dist)
    return top_ind, top_dist, dist

def evalImgSim(config, data_dl):
    """ Evaluate a pretrained model using test data """

    emd = EvalMMDEmbed(config)
    src_emb, src_fn    = emd.genImgEmb(data_dl)
    #print(src_emb.shape, src_fn)
    print(src_emb)
    #for simple test,  compare the images in the source list
    target_emb = src_emb
    target_fn = src_fn
    topk_ind, topk_dist = getKNN(src_emb, target_emb, topk=5)      
     
    sim_dict = {}
    for j, f in enumerate(src_fn):
       sim_dict[f] = {"knn":[], "knn_dist": []}
       sim_dict[f]["knn"] = [target_fn[i] for i in topk_ind[j,:].tolist()]
       sim_dict[f]["knn_dist"] = topk_dist[j,:].tolist()
    
    with open(config.out_file, "w") as fp:
        json.dump(sim_dict, fp, indent=2, sort_keys=True)
     
    return  sim_dict

def genImgEmb(config, data_dl):
    """ Evaluate a pretrained model using test data """

    emd = EvalMMDEmbed(config)
    src_emb, src_fn    = emd.genImgEmb(data_dl)
    #print(src_emb.shape, src_fn)
    target_emb = []
    target_fn = []
    with open("lists/afeats.lst", "r") as fa:
       for afile in fa:
          afile = afile.strip().replace("After_VAD_48k", "avc_sim_model_emb")
          target_emb.append(np.load(afile))
          target_fn.append(afile)
    target_emb = np.squeeze(np.asarray(target_emb))
    print("aud emb", target_emb.shape, len(target_fn))
    topk_ind, topk_dist = getKNN(src_emb, target_emb, topk=5)      
    s = np.squeeze(np.asarray([src_emb[0]]*len(target_fn)))
    print(s.shape)
    emd.compareEmb(s, target_emb)
     
    sim_dict = {}
    for j, f in enumerate(src_fn):
       sim_dict[f] = {"knn":[], "knn_dist": []}
       sim_dict[f]["knn"] = [target_fn[i] for i in topk_ind[j,:].tolist()]
       sim_dict[f]["knn_dist"] = topk_dist[j,:].tolist()
    
    print(sim_dict)   
    
    with open(config.out_file, "w") as fp:
        json.dump(sim_dict, fp, indent=2, sort_keys=True)
     
    return  sim_dict

def genAudEmb(config, data_dl):
    
    
    eval_model = EvalMMDEmbed(config)
    aud_embs, aud_list = gae.genAudEmb(config, data_dl, eval_model, save=False)
    return aud_embs, aud_list



def loadAudFileEmb(flist):
    """ Load pre-generated audio file-level embeddings """
    
    embs = None
    emb_fns =[]
    sub_ids = []
    for sf in flist:
        d = np.load(sf)
        x = d["arr_0"]
        ef = d["arr_1"].tolist()
        
        emb_fns.extend(ef)  
        sub, _ = os.path.splitext(sf.split("/")[-1]) #subcat ID
        sub = sub.replace("_files", "")
        sub_ids.extend([sub]*len(ef))   
        if x.ndim == 1:
           x = x.reshape(1, -1)
        if embs is None:
           embs = x
        else:
           embs = np.concatenate((embs,x), axis=0)
    #print(emb_fns[:2])    
    #return file-level embeddings and file names 
    return embs, emb_fns, sub_ids



def genAudioRec(config, data_dl, vis_data, aud_data):
    """ Get audio recommendations for visual frames using pretrained model""" 

    emd = EvalMMDEmbed(config)
    #generate the audio embeddings       
    aud_embs, aud_list = gae.genAudEmb(config, aud_data, emd)    
    vis_embs, vis_list    = emd.genImgEmb(data_dl)
    #print(vis_embs.shape, len(vis_list), vis_list[:5])
    #get the top-k audio recos for the visual frame
    topk_ind, topk_dist, dist = getKNN(vis_embs, aud_embs, config.topk+1)
    print(topk_ind.shape, topk_dist.shape)
    if not config.model.startswith("av_contrast"):
        va_cor = emd.compareAVEmb(vis_embs, aud_embs)

    #video to GT labels mapping
    vid2label = {}
    for n, v in enumerate(vis_data[0]):        
        vid2label[v.split("/")[0]] = vis_data[1][n]
    #audio to labels mapping
    aud2label = {}
    for n, v in enumerate(aud_data[0]):        
        aud2label[v] = aud_data[1][n]
        
        
    sim_dict = {}
    label_match = 0.0
    file_match = 0.0
    gt_cor = 0.0
    for j, f in enumerate(vis_list): 
       sim_dict[f] = {"knn":[], "knn_dist": [], "aud_labels": []}
       aud_fn = [os.path.splitext(aud_list[i].split("/")[-1])[0] for i in topk_ind[j,:].tolist()]
       #aud_fn = [os.path.splitext(aud_list[i].split("/")[-1])[0] for i in va_cor[j]["filtered_cor_ind"]]
       #top_corr = [va_cor[j]["cor"][i] for i in topk_ind[j,:].tolist()]
       #print("top_cor_ind", va_cor[j]["filtered_cor_ind"])
       #print("top_cor_dist", va_cor[j]["filtered_cor_dist"])

       sim_dict[f]["knn"] = aud_fn
       sim_dict[f]["knn_dist"] = topk_dist[j,:].tolist()
       #sim_dict[f]["knn_dist"] = va_cor[j]["filtered_cor_dist"]
       sim_dict[f]["gt_label"] = vis_data[1][j]
       sim_dict[f]["aud_labels"] = [aud2label[v] for v in aud_fn]
      
       if not config.model.startswith("av_contrast"):
           sim_dict[f]["gt_dist"] = float(va_cor[j]["dist"][j])
           sim_dict[f]["gt_cor"] = float(va_cor[j]["cor"][j])
       
           gt_cor += sim_dict[f]["gt_cor"]
       if sim_dict[f]["gt_label"] in sim_dict[f]["aud_labels"]:
           label_match += 1.0
       if f.split("/")[0] in sim_dict[f]["knn"]:
           file_match += 1.0    
       
    with open(config.out_file, "w") as fp:
        json.dump(sim_dict, fp, indent=2, sort_keys=True)   
    
    
    print("label accuracy=%f, file accuracy=%f, aud correlation accuracy=%f\n"%(label_match/len(vis_list), file_match/len(vis_list), gt_cor/len(vis_list))) 
    return 
    


    
  
    
    
