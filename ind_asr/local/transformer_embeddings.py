import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from espnet.utils.training.batchfy import make_batchset
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
import json
import logging
import kaldiio


root = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr"
with open(root+"/dump/nisp_test/deltafalse/data_with_labels.json", "r") as f:
    train_json = json.load(f)["utts"]

# train = {}
# x = list(train_json.items())
# for i in range(len(x)):
#     key,info = x[i]
#     if (info["category"]=="NPTEL"):
#         train[key] = (train_json[key])
train = train_json

##Load a pretrained model
asr_dir = root+"/exp/train_NPTEL_IITM_pytorch_final/results"
with open(asr_dir + "/model.json", "r") as f:
    idim, odim, conf = json.load(f)

transformer = E2E.build(idim, odim, **conf)
transformer.load_state_dict(torch.load(asr_dir + "/model.loss.best"))
transformer.cpu().eval()

embed_dict = {}
embed_json = {}
embed_dir = root+"/dann/nisp/embeddings/test/"
for key, info in train.items():
    fbank = torch.as_tensor(kaldiio.load_mat(info["input"][0]["feat"]))
    with torch.no_grad():
        embedding = transformer.encode(fbank)
        embed = embedding.clone().detach()
    embed_dict[key] = embed
    new_info = info
    new_info["input"][0]["feat"] = key
    new_info["input"][0]["shape"] = embed.size()
    embed_json[key] = new_info
# torch.save(embed_dict,embed_dir+"train_embeddings.pt")
torch.save(embed_dict,embed_dir+"test_embeddings.pt")
with open('/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/nisp/test_embed.json', 'w') as fp:
    json.dump(embed_json, fp, sort_keys=True, indent=4)