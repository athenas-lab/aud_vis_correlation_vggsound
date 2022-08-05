import os
import sys
import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import random



""" 
Data loader to load audio embeddings 
"""    



def _collate_fn(batch):
   """
   batch: ((data, subcat_idx, num_wins), ...)  for enroll
   batch: ((data, file_idx, num_wins), ...)  for test
   data: (num_wins, 1, win_len, 64)
   """
   transposed = list(zip(*batch))
   data = torch.cat(transposed[0], dim = 0)
   idx = np.array(transposed[1])
   num_wins = np.array(transposed[2])

   return data, idx, num_wins



def sliding_window(data, size = 100, stepsize=100, axis=0):
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )


    shape = list(data.shape)
    shape[axis] = int((data.shape[axis] - size) / stepsize + 1)
    shape[1] = size
    shape.append(data.shape[-1])

    strides = list(data.strides)
    strides.insert(0, strides[0] * stepsize)

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )


    return strided


class TestDataset(Dataset):
  def __init__(self, data_list, winlen = 100, stride = 100, aud_len=100):
      """
      data_list is a list of file paths
      """
      self.data_list = data_list
      self.win_len = winlen
      self.stride = stride
      self.aud_len = aud_len

  def __getitem__(self, idx):
      file_path = self.data_list[idx]
      feats = np.load(file_path)
      num_frms = feats.shape[0]
      #print("before padding", file_path, num_frms)
      if num_frms < self.win_len:
         feats = np.pad(feats, pad_width = ((0, self.aud_len - num_frms), (0, 0)), mode = 'wrap')
         num_frms = feats.shape[0]
#      else:
#         feats = feats[:self.aud_len]
      
     
      #avg of first k sec, where k is [0,5]
      avg_5sec = np.zeros((self.aud_len, feats.shape[1]))
      k = num_frms // self.aud_len    
      cnt = 0.0
      for i in range(min(5, k)):
         avg_5sec += feats[i*self.aud_len: (i+1)* self.aud_len, :]
         cnt +=1.0 
      avg_5sec /= cnt   
      feats = avg_5sec
      #print("after padding:before sliding window", num_frms, k, cnt, feats.shape)
      
      #changed from feats to avg_5sec
      feats = sliding_window(feats, size = self.win_len, stepsize = self.stride, axis = 0)
      
      feats=np.transpose(feats, axes=(0, 2, 1))   

      #print("after sliding window and transpose:{}:{}".format(file_path, feats.shape))
      feats = torch.FloatTensor(feats[:, np.newaxis, :, :])
      #print("====return value:{}:{},{},{}====".format(file_path, feats.shape, idx, feats.size(0)))
      return feats, idx, feats.size(0)

  def __len__(self):
      return len(self.data_list)



def get_loader_test(data_list, win_len = 100, stride = 100, batch_size = 16, aud_len=100):
   dset = TestDataset(data_list, win_len, stride, aud_len)
   loader = DataLoader(dset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = False, num_workers = 0, pin_memory = False)

   return loader

