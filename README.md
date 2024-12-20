# [Learning Self-Supervised Audio-Visual Representations for Sound Recommendations](https://arxiv.org/abs/2412.07406)


The code in this repo implements a self-supervised audio-visual correlation
model based on contrastive learning. The model has been trained and 
evaluated on VGGSound dataset. This model can be used to recommend sounds 
correlated with the objects/scenes in video frames/images.


# Data Preparation

* download the vggsound dataset
* generate visual  frames from the video (using ffmpeg, at 1 frame per second).
* generate corresponding audio frames (using ffmeg)    
* generate filterbank audio features for each audio sample (using librosa).
* create train/test list from the vggsound dataset. See the files in the misc/ folder. Sample lists are in the lists/ folder.

# Model Training

* modify the paths and configuration, if needed in config.py
* modify the path to the vggsound video frames and audio data in the files in the /data folder. This folder has the data loader and dataset implementation for vggsound.
* the models are in the models/folder. The model is a 2-stream deep convolutional model enhanced with multiple attention layers. It takes as input a pair of (image, filterbank audio feature generated from the audio). 
Audio and visual input that are temporally aligned in the 
original video form a positive pair while audio and visual input from different
time segments form a negative pair. The model is trained using contrastive loss.

* To train, run: python main_vggsound.py

# Model Evaluation

* modify the paths and configuration, if needed in eval_config.py. 
  * set the path to the trained model in this file.
  * provide the audio list  and visual frames for which
    sound recommendations need to be generated. See the files in the lists/ folder.
  * provide the path to the output file         
* To evaluate, run: python main_eval_reco.py

   * the evaluation code is in eval_self_super_av.py.
   * to generate sound recommendations, the code first generates the audio
and visual embeddings for the input using the trained contrastive model
and finds the top-k audio samples that 
are best correlated with the visual input, based on euclidean distance.
* the top-k sound recommendations for input image/frame will be generated as a 
  json file.






