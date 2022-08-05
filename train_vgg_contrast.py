import time
import os
import copy
import sys
import logging

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import torch
import torchvision.utils as utils
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn.parallel import data_parallel as dpar
from tensorboardX import SummaryWriter


sys.path.append("models")
import init_model as im
import att_utils as au
 
""" Trains the Self-supervised attention audio-Video correspondence models 
    using contrastive learning.
"""

def getModel(model, config):
     #if use_cuda:
           #  gpu_dtype = torch.cuda.FloatTensor
           #  self.model = self.model.type(gpu_dtype)
    use_cuda = torch.cuda.is_available() # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu") # use GPU or CPU
    num_gpus = torch.cuda.device_count()
    gpu_list = None
    if config.mgpu and num_gpus > 0: #use multi-gpus
    #if num_gpus > 0: 
             print("Using %d gpus"%num_gpus)
             gpu_list = list(range(num_gpus))
             model = torch.nn.DataParallel(model, device_ids=gpu_list)
    model.to(device) 
    return model, device, gpu_list

class TrainEvalAVC():

  def __init__(self, config):
     
        self.config = config
        self.nb_classes = config.num_classes

        self.writer = SummaryWriter(config.writer_out)
        self.images_disp =[]
        self.aud_disp = []
         
        #initialize the model, loss fn, and optimizer
        logging.info("Initializing model=%s"%config.model)
        self.model, self.loss_fn = im.initModel(self.config)
        self.model_params = self.model.parameters()
        #print(self.model_params)  
        self.initOptim()
        
        if self.model:
           self.model, self.device, self.gpu_list  = getModel(self.model, self.config) 


  def initOptim(self):
      """ Configure the optimizer """
      
      #SGD achieves better generalization than Adam

      #self.optim = optim.Adam(self.model_params, lr=self.config.learn_rate, weight_decay=1e-6)   
      #self.optim = optim.Adam(self.model_params, lr=self.config.learn_rate, weight_decay=5e-4)   
      self.optim = optim.SGD(self.model_params, lr=self.config.learn_rate, momentum=0.9, weight_decay=1e-4, nesterov=True) 
      #self.optim = optim.RMSprop(self.model_params, lr=self.config.learn_rate, momentum=0.9, weight_decay=5e-4)
      
      return 
  
  def adjustLearnRate(self, lr):
      """ Adjust the learning rate during training """

      for pg in self.optim.param_groups:
          pg["lr"] = lr
      return    

  
  def getLoss(self, logits, vemb, aemb, y_lab, av_dist=None, neg_aemb=None):
      """ Return the batch loss using the specified loss function """
      
      config = self.config
      batch_loss = im.getLoss(config, self.loss_fn, self.device, logits, vemb, aemb, y_lab, av_dist, neg_aemb)
      if config.reg_lambda > 0.0: #regularization
          l2_reg =  None
          for W in self.model.parameters():
              if l2_reg is None:
                  l2_reg = W.norm(2)
              else:
                  l2_reg = l2_reg + W.norm(2)
          reg_loss =  0.5 * (config.reg_lambda * l2_reg)          
          print(batch_loss, reg_loss)         
          batch_loss = batch_loss +  reg_loss    
      
      return batch_loss

 

 
  """ Baseline """  
  def trainEpoch(self, loader, mode):
      """ Run a single epoch in train mode """
    
      #set the model's mode
      self.model.train() 
     

      running_loss = 0.0
      running_acc = 0.0
      running_prec = 0.0
      running_rec = 0.0
      running_fscore = 0.0
      num_batches = 0
      epoch_samples = 0
      
      #print("data size", len(loader.dataset))
      #iterate over batches
      for batch_id,  sample in enumerate(loader):
          num_batches += 1
         
          #print (sample["x"].shape)
          #print("data size before pairwise extension", len(sample))
          #input format: <x:visual data, y:corresponding audio, y_neg: unrelated audio>
          #converted format: <x, y, 1>, <x,y_neg,0>
          v_dev = sample["vis"].to(self.device,  dtype=torch.float) #visual feats
          a_dev = sample["aud"].to(self.device,  dtype=torch.float) #audio  feats
         
          if batch_id == 0: #log sample images for tensorboard display
              self.images_disp.append(v_dev[:4,:,:,:])
              self.aud_disp.append(a_dev[:4,:,:,:])    
              
          #print("batch size after extension: v={}, a={}, labs={}".format(v_dev.size(), a_dev.size(), y_dev.size()))
          #total samples to compute mean
          epoch_samples += v_dev.shape[0] 
      
          #zero the gradient buffers before backprop
          self.optim.zero_grad()
          #track history in train mode
          with torch.set_grad_enabled(mode == "train"):
               #forward pass: get output and compute loss
              
               #model returns: vis_emb, vis_proj, aud_emb, aud_proj
               vemb, vproj, aemb, aproj = self.model(v_dev, a_dev)
    
               #print("y_dev", y_dev)
               #print("logits", logits)
               #logits, _, _, _ = dpar(self.model, inputs=(v_dev, a_dev), device_ids=self.gpu_list)
               #repurpose the parameters for contrastive loss
               batch_loss = self.getLoss(logits=vproj, vemb=vemb, aemb=aemb, y_lab=aproj)                
               #backprop in train mode
               batch_loss.backward() # backprop
               self.optim.step() #param update
               
               #update batch loss and accuracy
               running_loss += batch_loss.item() * vemb.shape[0]
               if (batch_id) % 100 == 0: #log once every 100 batches 
                  logging.info("Batch {}: loss = {}".format(batch_id, batch_loss.item()))
                  
                
                   
      # Avg loss and accuracy for the entire dataset after an epoch
      epoch_perf = {}
      epoch_perf["loss"] = running_loss /epoch_samples 
      epoch_perf["acc"] = running_acc*1.0 /epoch_samples 
      epoch_perf["prec"] = running_prec/num_batches
      epoch_perf["rec"] =  running_rec/num_batches
      epoch_perf["fscore"] = running_fscore/num_batches
      logging.info("{}:Total num samples={}, Num_batches = {}, Epoch loss = {},  Accuracy = {}, Precision = {}, Recall = {}, Fscore = {}".format(
                mode.capitalize(), epoch_samples, num_batches, epoch_perf["loss"], epoch_perf["acc"],
                epoch_perf["prec"], epoch_perf["rec"], epoch_perf["fscore"]))
      return epoch_perf

               
  def valEpoch(self, loader):
         """ Run a single epoch in validation mode """
    
         #set the  mode
         self.model.eval()           
         num_batches = 0
         y_preds_all = []
         y_all = []
         running_loss =0.0
         epoch_perf = {}
         epoch_perf["eer"] = 0.0
         epoch_perf["acc"] = 0.0


         with torch.no_grad():
            #iterate over batches
            for batch_id,  sample in enumerate(loader):
              num_batches += 1
              
              v_dev = sample["vis"].to(self.device,  dtype=torch.float) #visual feats
              a_dev = sample["aud"].to(self.device,  dtype=torch.float) #audio  feats
   
             
              if batch_id == 0: #log sample images for tensorboard display
                 self.images_disp.append(v_dev[:4,:,:,:])
                 self.aud_disp.append(a_dev[:4,:,:,:])    
             
              #print("batch size after extension: v={}, a={}, labs={}".format(v_dev.size(), a_dev.size(), y_dev.size()))

              #forward pass: get output
              #model returns: vis_emb, vis_proj, aud_emb, aud_proj
              vemb, vproj, aemb, aproj = self.model(v_dev, a_dev)
    
              #print("y_dev", y_dev)
              #print("logits", logits)
              #logits, _, _, _ = dpar(self.model, inputs=(v_dev, a_dev), device_ids=self.gpu_list)
              #repurpose the parameters for contrastive loss, #avg loss for the batch
              batch_loss = self.getLoss(logits=vproj, vemb=vemb, aemb=aemb, y_lab=aproj)          
              running_loss += batch_loss.item() #*v_dev.shape[0]
              #print(logits, y_dev)
              
    
         epoch_perf["loss"] = running_loss /num_batches #y_all.shape[0]
         #epoch_perf["prec"], epoch_perf["rec"], epoch_perf["fscore"], _ = precision_recall_fscore_support(y_all, y_preds_all, average="micro")
                  
         logging.info("{}: Epoch loss = {}".format("Validation", epoch_perf["loss"]))
         return epoch_perf

          
  def trainModel(self, data_loaders):
        
      start_time = time.time()
      print("{}: Start model training".format(start_time) )
      logging.info("{}: Start model training".format(start_time))
      
      config = self.config
    
      #used to keep track of the best performance
      epoch_loss_hist = {"train": [], "val": []}
      epoch_acc_hist  = {"train": [], "val": []}
      epoch_eer_hist  = {"train": [], "val": []}
      #records best perf
      best = {}
      best["model_wts"] = copy.deepcopy(self.model.state_dict())
      best["ep"] = 0 
      best["acc"] = 0.0
      best["loss"] = np.inf
      best["eer"] = np.inf
      num_epochs = config.num_epochs
      #early stop counter
      early_stop =False
      es_cnt = 0
    
    
      #iterate over epochs
      for epoch in range(num_epochs):
            #self.scheduler.step()
            if early_stop:
               break
            logging.info("Epoch {}/{}".format(epoch, num_epochs-1))
            logging.info("-" * 10)
            self.writer.add_scalar('train/learning_rate', self.optim.param_groups[0]['lr'], epoch)
            self.images_disp = []
            self.aud_disp = []
             
            #set the appropriate mode
            mode = "train"
            #train a single epoch
            epoch_perf= self.trainEpoch(data_loaders[mode], mode)
            #epoch_perf= self.trainPosNegSepEpoch(data_loaders[mode], mode)
            epoch_loss_hist[mode].append(epoch_perf["loss"])
            epoch_acc_hist[mode].append(epoch_perf["acc"])
            self.writer.add_scalar('train/loss', epoch_perf["loss"], epoch)
            self.writer.add_scalar('train/accuracy', epoch_perf["acc"], epoch)
            
            #validate periodically
            if epoch % config.eval_every == 0:
               mode = "val"
               epoch_perf = self.valEpoch(data_loaders[mode])
               epoch_acc_hist[mode].append(epoch_perf["acc"])
               epoch_eer_hist[mode].append(epoch_perf["eer"])
               epoch_loss_hist[mode].append(epoch_perf["loss"])
               #tensorboard logs
               self.writer.add_scalar('val/loss', epoch_perf["loss"], epoch)
               self.writer.add_scalar('val/accuracy', epoch_perf["acc"], epoch)
               I_train = utils.make_grid(self.images_disp[0], nrow=4, normalize=True, scale_each=True)
               A_train = utils.make_grid(self.aud_disp[0], nrow=4, normalize=True, scale_each=True)
               self.writer.add_image('train/image', I_train, epoch)
               self.writer.add_image('train/aud', A_train, epoch)
               if epoch == 0:
                    I_test = utils.make_grid(self.images_disp[1], nrow=4, normalize=True, scale_each=True)
                    A_test = utils.make_grid(self.aud_disp[1], nrow=4, normalize=True, scale_each=True)
                    self.writer.add_image('test/image', I_test, epoch)
                    self.writer.add_image('test/aud', A_test, epoch)
                    
               if config.att and config.att_cfg["disp_att"]:
                   au.visualizeAttMaps(self.model, epoch, self.writer, self.images_disp, I_train, I_test, "vis") 
                   au.visualizeAttMaps(self.model, epoch, self.writer, self.aud_disp, A_train, A_test, "aud") 
                   
               #record the best model running_acc = 0.0
               #if epoch_perf["acc"] > best["acc"]: 
               if (epoch_perf["loss"] < best["loss"]): 
               #if epoch_eer < best_eer:     
                  #deep copy the model
                  es_cnt = 0 #reset the early_stop counter
                  best["acc"] = epoch_perf["acc"]
                  best["eer"] = epoch_perf["eer"]
                  best["loss"] = epoch_perf["loss"]
                  best["model_wts"] = copy.deepcopy(self.model.state_dict())
                  best["ep"] = epoch
                  torch.save(self.model.state_dict(), os.path.join(config.model_checkpt_dir, "best_model_wts.pth"))
                  torch.save(self.optim.state_dict(), os.path.join(config.model_checkpt_dir, "best_optim.pth"))
                  torch.save(self.model.state_dict(), os.path.join(config.model_checkpt_dir, "best_model_wts_%d.pth"%epoch))
                  torch.save(self.optim.state_dict(), os.path.join(config.model_checkpt_dir, "best_optim_%d.pth"%epoch))
                  logging.info("Val perf improved: Epoch {}: Saved best model weights in {}".format(epoch, os.path.join(config.model_checkpt_dir, "best_model_wts.pth")))
                  print("Val perf improved: Epoch {}: Acc={}: Loss={}".format(epoch, best["acc"], best["loss"]))

               else: # no improvement. Check if early stopping criterion is satisfied
                   es_cnt += 1 
                   logging.info("No improvement: Best val accuracy:{}, Best val EER:{}, Best Loss: {} at epoch {}".format(best["acc"], 
                                best["eer"], best["loss"], best["ep"]))
                   if es_cnt  > config.early_stop:
                      early_stop = True 
                      logging.info("Early stopping at epoch %d"%epoch)
                      break 
                   if es_cnt % 5 == 0: #adapt LR every 5 epochs if there is no improvement
                      #new_lr = self.config.learn_rate/(epoch+1) 
                      #OTS paper
                      new_lr = self.config.learn_rate - 0.06*self.config.learn_rate
                      new_lr = max(0.0001, new_lr)
                      self.adjustLearnRate(new_lr)
                      logging.info("Adjusted learn rate to {}".format(new_lr))
         
                      
      torch.save(self.model.state_dict(), os.path.join(config.model_checkpt_dir, "final_model_wts.pth"))
      torch.save(self.optim.state_dict(), os.path.join(config.model_checkpt_dir, "final_optim.pth"))
 
      time_elapsed = (time.time() - start_time)/60.0
      print("Training Completed in {} minutes".format(time_elapsed))
      print("Best val acc = {}, Best Loss= {}: in epoch {}".format(best["acc"], best["loss"], best["ep"]))
      print("Saved best model weights in {}".format(os.path.join(config.model_checkpt_dir, "best_model_wts.pth")))
      logging.info("Training Completed in {} minutes".format(time_elapsed))
      logging.info("Best val acc = {}, Best Loss={}: in epoch {}".format(best["acc"], best["loss"], best["ep"]))
      logging.info("Saved best model weights in {}".format(os.path.join(config.model_checkpt_dir, "best_model_wts.pth")))
      #plt = mlm.plotLearning(epoch_loss_hist, epoch_acc_hist) # epoch_eer_hist)
      #plt_file = "%s.png"%(os.path.join(config.result_dir, "learn_curves"))
      #plt.savefig(plt_file, dpi=600) 

      #load the best model weights
      self.model.load_state_dict(best["model_wts"])
      return self.model
  
  

   
def trainModel(config,  dl_dict):
    """ Train a model and evaluate it  using test data """

    tea = TrainEvalAVC(config)
    best_model = tea.trainModel(dl_dict)

    return 

