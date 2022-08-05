import json


def createNegLabels():
    neg_label = {}
    with open("lists/labels2vid.json", "r") as fp:
        labs2vid = json.load(fp)
        print(len(labs2vid.keys()))
        labs = labs2vid.keys()
        for lab in labs:
            val = set(labs) - set([lab])
            neg_label[lab] = list(val)
            
    with open("lists/labels2neg.json", "w") as fout:
        json.dump(neg_label, fout, sort_keys=True, indent=2)


def createPosNegPairs(inp_file, out_file):
    """ Create the positive and negative AV pairs for training """

    #load the labels to video mapping
    fp =  open("lists/labels2vid.json", "r")
    labs2vid = json.load(fp)
    
    vid_to_pos_neg = {}
    vid_list = []
    lab_list = []
    with open(inp_file, "r") as ft:
        #compile a list of videos and labels
        for line in ft:
            vid, label = line.strip().split("|")
            vid_list.append(vid)
            lab_list.append(label)
        #for each video, get the list of positive and negative audio 
        #in the corresponding split    
        for i, v in enumerate(vid_list):
           label = lab_list[i]
           pos_list = [] #list of pos audio
           neg_list = [] #list of neg audio
           for vid in labs2vid[label]:
                    #filter only the videos that are in the input (train/test) split
                    if vid in vid_list:  
                       pos_list.append(vid)
           neg_list = list(set(vid_list) - set(pos_list))       
           vid_to_pos_neg[v] = {}
           vid_to_pos_neg[v]["label"] = label
           vid_to_pos_neg[v]["pos_aud"] = pos_list
           vid_to_pos_neg[v]["neg_aud"] = neg_list

    #output the video to audio mappings         
    with open(out_file, "w") as fout:
        json.dump(vid_to_pos_neg, fout, sort_keys=True, indent=2)

    return  

def processMeta():  
    multilabels = []
    unique = []
    md_path = "/data/vggsound.csv"
    with open(md_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if "\"" in line:
                print(line)
                multilabels.append(line)
                label = ";".join(line.split(",")[2:-1])
                unique.append(label)
    unique = list(set(unique))
    print(unique, len(unique))            

if __name__ == "__main__":
  #processMeta()  
  #createNegLabels()
  #createPosNegPairs("lists/vggsound_train.txt", "lists/vid2aud_train.json")
  createPosNegPairs("lists/vggsound_test.txt", "lists/vid2aud_test1.json")
  #createPosNegPairs("lists/vggsound_val.txt", "lists/vid2aud_val.json")
