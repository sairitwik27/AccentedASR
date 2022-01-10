#!/usr/bin/env python3

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from batchfy import make_batchset
# from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import json
import logging
# import kaldiio
from sklearn.metrics import ConfusionMatrixDisplay as CMD
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from acc_classifier import AccentClassifier

logging.basicConfig(filename="/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_nisp/classifier_test.log",filemode='a',level=logging.INFO)

def test(network, testloader):
    all_preds = torch.tensor([], device=device)
    all_labels= torch.tensor([], device=device)

    correct = 0
    total = 0
    model.eval()
    
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs,_ = network(inputs,labels)
        _, preds = torch.max(outputs.data, 1)
        all_preds = torch.cat((all_preds, preds),dim=0)
        all_labels = torch.cat((all_labels, labels),dim=0)
    return all_preds,all_labels

out_dir = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_nisp/"
root = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr"
with open(root + "/dump/nisp_test/deltafalse/data_with_labels.json", "r") as f:
    test_json = json.load(f)["utts"]

val_embed_dir = root+"/dann/nisp/embeddings/test/"
val_embed_dict = torch.load(val_embed_dir+"test_embeddings.pt")
batch_size = 32
testset = make_batchset(test_json, batch_size)

##Load a pretrained model
# asr_dir = root+"/exp/train_NPTEL_IITM_pytorch_final/results"
# with open(asr_dir + "/model.json", "r") as f:
#     idim, odim, conf = json.load(f)

# transformer = E2E.build(idim, odim, **conf)
# transformer.load_state_dict(torch.load(asr_dir + "/model.loss.best"))
# transformer.cpu().eval()


# def classifier_collate(minibatch):
#     feats = []
#     tokens = []
#     accent_labels = []
#     for key, info in minibatch[0]:
#         # fbank = torch.as_tensor(kaldiio.load_mat(info["input"][0]["feat"]))
#         feat = train_embed_dict[key]
#         # with torch.no_grad():
#         #     embedding = transformer.encode(fbank)
#         # feats.append(embedding.clone().detach())
#         feats.append(feat)
#         accent_labels.append(torch.tensor(int(info["acc_label"])) if info["category"]!='NPTEL' else torch.tensor(int(100)))
#     return pad_sequence(feats, batch_first=True),torch.tensor(accent_labels)

def classifier_collate(minibatch):
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

classifier = AccentClassifier(5)
model = classifier
device = torch.device("cuda")
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
checkpoint = torch.load("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier_nisp/snapshot.ep.23")
model.load_state_dict(checkpoint['state_dict'])
test_loader = DataLoader(testset, collate_fn=classifier_collate, pin_memory=True)

with torch.no_grad():
    all_preds, all_labels = test(model.to(device), test_loader)

logging.info(f'Accuracy:{accuracy_score(all_labels.cpu(), all_preds.cpu())}')
cmt= confusion_matrix(all_labels.cpu(), all_preds.cpu())
logging.info(cmt)