import json
import argparse

def append_labels_IITM(train_json):
    x = list(train_json['utts'].items())
    for i in range(len(x)):
        key,info = x[i]
        place = info["utt2spk"][:3]

        if (place.isdigit()):
            info["acc_label"] = 100

        if (place=='ahd'):
            info["acc_label"] = 0

        elif (place=='blr'):
            info["acc_label"] = 1

        elif (place=='ch_'):
            info["acc_label"] = 2

        elif (place=='hyd'):
            info["acc_label"] = 3

        elif (place=='bbs'):
            info["acc_label"] = 4

        elif (place=='dli'):
            info["acc_label"] = 5

        elif (place=='kol'):
            info["acc_label"] = 6

        elif (place=='lnw'):
            info["acc_label"] = 7

        elif (place=='mum'):
            info["acc_label"] = 8

        elif (place=='pue'):
            info["acc_label"] = 9


def append_labels_NISP(train_json):
    x = list(train_json['utts'].items())
    for i in range(len(x)):
        key,info = x[i]
        place = info["utt2spk"][:-5]


        if (place=='Hindi'):
            info["acc_label"] = 0

        elif (place=='Kannada'):
            info["acc_label"] = 1

        elif (place=='Malayalam'):
            info["acc_label"] = 2

        elif (place=='Tamil'):
            info["acc_label"] = 3

        elif (place=='Telugu'):
            info["acc_label"] = 4

def append_cat_IITM(train_json):
    x = list(train_json['utts'].items())
    for i in range(len(x)):
        key,info = x[i]
        place = info["utt2spk"][:3]
        if ((place.isdigit())):
            info["category"] = 'NPTEL'
        else:
            info["category"] = 'IITM'


parser = argparse.ArgumentParser(description='Append Labels')
parser.add_argument("-i","--inpath", help="Give the path to input json", default="/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dump/train_NPTEL_IITM/deltafalse/data.json")
parser.add_argument("-o","--outpath", help="Give the path to output json", default="/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dump/train_NPTEL_IITM/deltafalse/data_with_labels.json")

args = parser.parse_args()

with open(args.inpath, "r") as f:
    train_json = json.load(f)
    
# append_cat_IITM(train_json)
# append_labels_IITM(train_json)
append_labels_NISP(train_json)

with open(args.outpath,"w") as f:
    json.dump(train_json, f,indent=4)
