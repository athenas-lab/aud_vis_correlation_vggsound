# Self-supervised audio-visual correlation model


The code in this repo implements a self-supervised audio-visual correlation
model based on contrastive learning. The model has been trained and 
evaluated on VGGSound dataset. This model can be used to recommend sounds 
correlated with the objects/scenes in video frames/images.

# To train the model:

* modify the paths and configuration, if needed in config.py
* modify the path to the vggsound video frames and audio data in the files in the /data folder. This folder has the data loader and dataset implementation for vggsound.

# To evaluate the model

* modify the paths and configuration, if needed in eval_config.py. 
  * Set the path to the trained model in this file.
  * provide the list of audio and visual frames for which
    sound recommendations need to be generated.
  * provide the path to the output file         
* the top-k sound recommendations for each image/frame will be generated as a 
  json file.





