import os 
import json

""" Process VGGSound Metadata file  and generates the train/test split"""


md_path = "data/vggsound/vggsound.csv"  #change it to appropriate path

train_gt = {}
test_gt = {}
labels2vid = {} #reverse mapping for negative labels during training

def processMeta():
    labels = []
    train = 0
    test = 0
    available = []
    #get the list of available videos
    for x in os.listdir("data/vggsound/video"): #use the path for vggsound videos
       f, ext = os.path.splitext(x)
       available.append(f)
    #print("available", len(available))   
       
    with open(md_path, "r") as fin:
        for line in fin:
            line = line.strip()
            #video,timestamp,label,split
            parts = line.split(",")
            parts = [x.strip() for x in parts]
            #for vggsound, multilabels are synonyms. So we just use the first label.
            if len(parts) > 4:
                parts[2] = parts[2].replace("\"", "") #";".join(parts[2:-1])
            video = parts[0]+ "_" + parts[1]
            #map labels to videos
            if video in available:
             labels.append(parts[2])
             if parts[2] not in list(labels2vid.keys()):
               labels2vid[parts[2]] = [video]
             else:   
               labels2vid[parts[2]].append(video)
               
            if parts[-1] == "train":
               train+=1
               if video in available:
                  train_gt[video] = parts[2]
            elif parts[-1] == "test":
               test+=1
               if video in available:
                  test_gt[video] = parts[2]
    print("Total number of samples={}, num train={}, num test={}, num download={}".format(len(labels), train, test, len(available)))        
    #print(available)
    return labels


def outSplits(v_gt, outf):
    """ Output train/test splits """

    multi_labs = 0
    with open(outf, "w") as fout:
      for k,v in v_gt.items():
        fout.write("%s|%s\n"%(k,v))
        if ";" in v:
            multi_labs +=1
    print("num of multilabels=%d"%multi_labs)        
    return    

def outLabs2Vid(meta, outf):
    """ Output mapping from labels to videos """

    with open(outf, "w") as fout:
        json.dump(meta, fout, sort_keys=True, indent=2)
    print("num of labels",len(meta))    
    return    

if __name__ == "__main__":
   labs =  processMeta()
   labs = list(set(labs))
   print("number of unique labels={}, actual train={}, actual test={}".format(len(labs), len(train_gt), len(test_gt)))
   #print(list(train_gt.keys())[0], train_gt[list(train_gt.keys())[0]])
   print("unique train labels={}, unique test labels={}".format(len(list(set(train_gt.values()))), len(list(set(test_gt.values())))) )
   outSplits(train_gt, "lists/vggsound_train.txt")
   outSplits(test_gt, "lists/vggsound_test.txt")
   outLabs2Vid(labels2vid, "lists/labels2vid.json")
   







