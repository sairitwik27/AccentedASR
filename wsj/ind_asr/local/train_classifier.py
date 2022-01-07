#!/usr/bin/env python3

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from espnet.utils.training.batchfy import make_batchset
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import json
import logging
import kaldiio

logging.basicConfig(filename="/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_new/classifier_train_new.log",filemode='a',level=logging.INFO)

out_dir = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_new/"
root = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr"
with open(root + "/dump/train_NPTEL_IITM/deltafalse/data_with_labels.json", "r") as f:
    train_json = json.load(f)["utts"]
with open(root + "/dump/dev_IITM/deltafalse/data_with_labels.json", "r") as f:
    dev_json = json.load(f)["utts"]

train = {}
x = list(train_json.items())
for i in range(len(x)):
    key,info = x[i]
    if (info["category"]=="IITM"):
        train[key] = (train_json[key])

batch_size = 32
trainset = make_batchset(train, batch_size)
devset = make_batchset(dev_json, batch_size)

##Load a pretrained model
asr_dir = root+"/exp/train_NPTEL_IITM_pytorch_final/results"
with open(asr_dir + "/model.json", "r") as f:
    idim, odim, conf = json.load(f)

transformer = E2E.build(idim, odim, **conf)
transformer.load_state_dict(torch.load(asr_dir + "/model.loss.best"))
transformer.cpu().eval()


def classifier_collate(minibatch):
    feats = []
    tokens = []
    accent_labels = []
    for key, info in minibatch[0]:
        fbank = torch.as_tensor(kaldiio.load_mat(info["input"][0]["feat"]))
        with torch.no_grad():
            embedding = transformer.encode(fbank)
        feats.append(embedding.clone().detach())
        accent_labels.append(torch.tensor(int(info["acc_label"])) if info["category"]!='NPTEL' else torch.tensor(int(100)))
    return pad_sequence(feats, batch_first=True),torch.tensor(accent_labels)

train_loader = DataLoader(trainset, collate_fn=classifier_collate, shuffle=True, pin_memory=True)
dev_loader = DataLoader(devset, collate_fn=classifier_collate, pin_memory=True)


import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from acc_classifier import AccentClassifier

load = True
classifier = AccentClassifier(10)
model = classifier
device = torch.device("cuda")
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))

if load:
    checkpoint = torch.load("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_new/snapshot.ep.3")
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

n_epoch = 30
best_loss = 10000
logging.info("Beginning to Train!")
for epoch in range(3,n_epoch+1):
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
