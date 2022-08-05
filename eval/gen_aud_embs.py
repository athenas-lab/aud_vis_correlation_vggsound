import os 
import sys
import time
import json
from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances as ed

import torch
from torch.nn.parallel import data_parallel as par
import numpy as np

from aud_loader import get_loader_test

""" Generate audio embeddings from trained AV correlation models """


#path to the filterbank features for the VGGSound audio files. 
#These features need to be generated.
aud_root = "dataset/vggsound/audio_feats/fb_48k"

sys.path.append("models")


def readmv():
    """ Read precomputed mean and variance """

    params = {}
    mv = np.load("../lists/vggsound_train_mean_var.npy") 
    params["mean"] = torch.FloatTensor(mv[0])
    eps = 1e-8
    params["std"] = torch.FloatTensor(np.sqrt(mv[1]) + eps)
    return params


def outEmbForFile(emb, emb_file):
    """ Save audio embeddings to file """

    # output embedding path, change this to yours
    emb_out = os.path.join(conf.aud_embs, "/".join(emb_file.split("/")[8:])) 
    par_dir = os.path.dirname(emb_out)
    if not os.path.exists(par_dir):
       os.makedirs(par_dir) 
    np.save(emb_out, emb)
    print('{} generated: {}'.format(emb_out, emb.shape))
    return

# Generate audio level embedding
def gen_emb_wav(model, global_mean, global_std, device, feat_list, win_len = 100, aud_len=100, stride = 100, batch_size = 256):

    loader = get_loader_test(feat_list, win_len, stride, batch_size, aud_len)
    aud_embs = []
    aud_list = []
    for x, file_ids, num_wins in loader:
           #print("xshape, file_ids, num_wins", x.shape, file_ids, num_wins)
           x = torch.transpose(x, dim0=2, dim1=3)
           #print("before transpose:", global_mean.shape, global_std.shape, x.shape)
           x -= global_mean
           x /= global_std
           x = torch.transpose(x, dim0=2, dim1=3)
           #print("after transpose:", global_mean.shape, global_std.shape, x.shape)

   
           x = x.to(device, dtype=torch.float)
           start_idx = 0
           i = 0
           with torch.no_grad():
               #print(x.shape)
               seg_embs = model.module.getAudEmb(x)
               seg_embs = seg_embs.data.cpu().numpy()
               #print("seg_embs shape", seg_embs.shape)

               for num_win in num_wins:
                   utt_emb = np.mean(seg_embs[start_idx : start_idx + num_win], axis = 0, keepdims = True)  # audio level embedding
                   n = np.linalg.norm(utt_emb, ord = 2, axis = 1, keepdims = False) # normalize embedding 
                   utt_emb /= (n + 1e-8)
                   
    
                   file_id = file_ids[i]
                   in_path = feat_list[file_id]   # input feature file path
                   #print("embedding shape={}, for feature file={}".format(utt_emb.shape, in_path))
                   aud_embs.append(utt_emb)
                   aud_list.append(in_path)
                   start_idx += num_win
                   i += 1
                   
    aud_embs = np.squeeze(np.asarray(aud_embs)) 
    if aud_embs.ndim == 1:
       aud_embs = aud_embs.reshape(1,-1)             
    #print(feat_list==aud_list, aud_embs.shape)       
    return aud_embs, aud_list


    
def compDist(aud_embs, aud_list, labels):
    """ Test the embeddings by computing distance between 
        audio from same and different categories """

    d = ed(aud_embs, aud_embs)
    ids = np.argsort(d, axis=1)
    #print("distance", d, ids)
    sim_labels = []
    match = 0
    for i in range(ids.shape[0]):
        #get labels of top-10 similar aud nbrs
        nbrs = [labels[j] for j in ids[i,:]][:11]
        sim_labels.append(nbrs)
        if nbrs[0] in nbrs[1:]:
            match += 1
    #print("compDist:sim labels={}, label_accuracy={}".format(sim_labels, match*1.0/len(aud_list)))    
    return

def loadEmbs(sources):
    """ Load embeddings from file """

    embs = []
    efs = []
    for s in sources:
           embs.append(np.load(s))
          
    embs = np.squeeze(np.asarray(embs))
    if embs.ndim == 1:
       embs = embs.reshape(1,-1)
    return embs, efs



def genAudEmb(config, data_list, em, save=False):
    
   start = time.time()
   print(start)
   #read the training data  mean and var
   params  = readmv() 
   train_mean, train_std = params["mean"], params["std"]
   feat_list = []
   
   # generate audio level embedding for these feature files 
   for afile in data_list[0]:
         
         afile = os.path.join(aud_root, afile)+".npy"
         feat_list.append(afile)

   aud_embs, aud_list = gen_emb_wav(em.model, train_mean, train_std, em.device, feat_list, win_len=config.win_len, aud_len = config.aud_frames, stride = 50, batch_size =64)  # audio level embedding
   print("time to generate emb", time.time() - start)
   compDist(aud_embs, aud_list, data_list[1])
   return aud_embs, aud_list

  
