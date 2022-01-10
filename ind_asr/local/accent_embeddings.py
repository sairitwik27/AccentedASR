import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from espnet.utils.training.batchfy import make_batchset
import json
import logging
import kaldiio
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from acc_classifier import AccentClassifier

classifier = AccentClassifier(10)
model = classifier
device = torch.device("cuda")
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
checkpoint = torch.load("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_train/snapshot.ep.50")
model.load_state_dict(checkpoint['state_dict'])
model.cpu().eval()

root = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr"
# with open(root + "/dump/train_NPTEL_IITM/deltafalse/data_with_labels.json", "r") as f:
#     train_json = json.load(f)["utts"]

with open(root + "/dump/train_NPTEL_IITM/deltafalse/data_with_labels.json", "r") as f:
    train_json = json.load(f)["utts"]

train_NPTEL = {}
train_IITM = {}
x = list(train_json.items())
for i in range(len(x)):
    key,info = x[i]
    if (info["category"]=="NPTEL"):
        train_NPTEL[key] = (train_json[key])
    if (info["category"]=="IITM"):
        train_IITM[key] = (train_json[key])

del train_json


NPTEL_embed_dir = root+"/dann/NPTEL/embeddings/train/"
NPTEL_embed_dict = torch.load(NPTEL_embed_dir+"train_embeddings.pt")

embed_dict = {}
embed_json = {}
embed_dir = root+"/dann/acc_embeddings/train/"

for key, info in train_NPTEL.items():
    feat = NPTEL_embed_dict[key]
    feat = torch.as_tensor(feat).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode(feat)
        embed = embedding.clone().detach()
    embed_dict[key] = embed
    new_info = info
    new_info["input"][0]["feat"] = key
    new_info["input"][0]["shape"] = embed.size()
    embed_json[key] = new_info

del NPTEL_embed_dict

IITM_embed_dir = root+"/dann/IITM/embeddings/train/"
IITM_embed_dict = torch.load(IITM_embed_dir+"train_embeddings.pt")
for key, info in train_IITM.items():
    feat = IITM_embed_dict[key]
    feat = torch.as_tensor(feat).unsqueeze(0)
    with torch.no_grad():
        embedding = classifier.encode(feat)
        embed = embedding.clone().detach()
    embed_dict[key] = embed
    new_info = info
    new_info["input"][0]["feat"] = key
    new_info["input"][0]["shape"] = embed.size()
    embed_json[key] = new_info
del IITM_embed_dict
torch.save(embed_dict,embed_dir+"train_embeddings.pt")
# torch.save(embed_dict,embed_dir+"val_embeddings.pt")
with open('/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/acc_embeddings/train_embed.json', 'w') as fp:
    json.dump(embed_json, fp, sort_keys=True, indent=4)