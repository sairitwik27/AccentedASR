#!/usr/bin/env python3

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from batchfy import make_batchset
# from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import json
import logging
# import kaldiio

out_dir = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_nisp/"
logging.basicConfig(filename=out_dir+"classifier_train_new.log",filemode='a',level=logging.INFO)
root = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr"


with open("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/nisp/train_embed.json", "r") as f:
    train_json = json.load(f)
with open(root + "/dann/nisp/test_embed.json", "r") as f:
    dev_json = json.load(f)

train_embed_dir = root+"/dann/nisp/embeddings/train/"
val_embed_dir = root+"/dann/nisp/embeddings/test/"
train_embed_dict = torch.load(train_embed_dir+"train_embeddings.pt")
val_embed_dict = torch.load(val_embed_dir+"test_embeddings.pt")
train = {}
x = list(train_json.items())
# for i in range(len(x)):
#     key,info = x[i]
#     if (info["category"]=="IITM"):
#         train[key] = (train_json[key])

batch_size = 32
trainset = make_batchset(train_json, batch_size)
devset = make_batchset(dev_json, batch_size)

##Load a pretrained model
# asr_dir = root+"/exp/train_NPTEL_IITM_pytorch_final/results"
# with open(asr_dir + "/model.json", "r") as f:
#     idim, odim, conf = json.load(f)

# transformer = E2E.build(idim, odim, **conf)
# transformer.load_state_dict(torch.load(asr_dir + "/model.loss.best"))
# transformer.cpu().eval()


def classifier_collate_train(minibatch):
    feats = []
    # tokens = []
    accent_labels = []
    for key, info in minibatch[0]:
        # feat = embed_dict[info["input"][0]["feat"]]
        feat = train_embed_dict[key]
        # fbank = torch.as_tensor(kaldiio.load_mat(info["input"][0]["feat"]))
        # with torch.no_grad():
            # embedding = transformer.encode(fbank)
        feats.append(feat)
        accent_labels.append(torch.tensor(int(info["acc_label"])))
    return pad_sequence(feats, batch_first=True),torch.tensor(accent_labels)

def classifier_collate_val(minibatch):
    feats = []
    # tokens = []
    accent_labels = []
    for key, info in minibatch[0]:
        # feat = embed_dict[info["input"][0]["feat"]]
        feat = val_embed_dict[key]
        # fbank = torch.as_tensor(kaldiio.load_mat(info["input"][0]["feat"]))
        # with torch.no_grad():
            # embedding = transformer.encode(fbank)
        feats.append(feat)
        accent_labels.append(torch.tensor(int(info["acc_label"])))
    return pad_sequence(feats, batch_first=True),torch.tensor(accent_labels)

train_loader = DataLoader(trainset, collate_fn=classifier_collate_train, shuffle=True, pin_memory=True)
dev_loader = DataLoader(devset, collate_fn=classifier_collate_val, pin_memory=True)

logging.info("Data Loaded!")

import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from acc_classifier import AccentClassifier

load = False
classifier = AccentClassifier(5)
model = classifier
device = torch.device("cuda")
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))

logging.info("Model Loaded!")

resume_epoch = 1
if load:
    resume_epoch = 50
    checkpoint = torch.load("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_tr/snapshot.ep."+str(resume_epoch))
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

n_epoch = 30
best_loss = 10000
logging.info("Beginning to Train!")
for epoch in range(resume_epoch,n_epoch+1):
    n_iter = 0
    loss_rec = []
    model.train()
    for data in train_loader:
        predictions,loss = model(*[d.cuda() for d in data])
        optim.zero_grad()
        loss.backward()
        loss_rec.append(model.loss)
        norm = clip_grad_norm_(model.parameters(), 10.0)
        optim.step()
        n_iter+=1
        if(n_iter%200==0):
            iter_loss = torch.mean(torch.stack(loss_rec))
            #iter_loss = np.mean(loss_rec)
            logging.info(f"iter:{n_iter},mean_loss:{iter_loss}")

    train_loss = torch.mean(torch.stack(loss_rec))
    state = {'epoch':epoch,
             'state_dict':model.state_dict(),
             'optimizer':optim.state_dict(),
    }
    torch.save(state,out_dir+"snapshot.ep."+str(epoch))
    
    loss_rec = []
    model.eval()
    n_total = 0
    n_correct = 0
    for data in dev_loader:
        val_pred,val_loss = model(*[d.to(device) for d in data])
        loss_rec.append(val_loss)
    valid_loss = torch.mean(torch.stack(loss_rec))
    logging.info(f"epoch: {epoch}, train loss: {train_loss:.3f}, dev loss: {valid_loss:.3f}")
    
    if (valid_loss<best_loss):
       best_loss = loss
       torch.save(model.state_dict(),out_dir+"model.acc.best")