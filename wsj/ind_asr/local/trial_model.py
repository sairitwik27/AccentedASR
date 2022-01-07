import torch
import json
# from acc_classifier import AccentClassifier
# from DANN import GRL,DANN_ASR
# from espnet.utils.dynamic_import import dynamic_import
# from espnet.utils.deterministic_utils import set_deterministic_pytorch
# from dann_asr_train import get_parser

# cmd_args =  (["--config","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/conf/train_transfer.yaml",
#               "--preprocess-conf","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/conf/no_preprocess.yaml",
#               "--ngpu","0",
#               "--backend","pytorch",
#               "--outdir","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/exp/dann_asr/results",
#               "--tensorboard-dir","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/tensorboard/dann_asr",
#               "--debugmode","1",
#               "--dict","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/data/lang_1char/train_NPTEL_IITM_units.txt",
#               "--debugdir","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/exp/dann_asr",
#               "--minibatches","0",
#               "--verbose","0",
#               "--resume","",
#               "--seed","1",
#               "--train-json","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/train/data_with_labels.json",
#               "--valid-json","/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/val/data_with_labels.json"])



# valid_json = "/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/dann/val/data_with_labels.json"
# with open(valid_json, "rb") as f:
#     valid_json = json.load(f)["utts"]
# utts = list(valid_json.keys())
# num_encs = 1
# idim_list = [
#     int(valid_json[utts[0]]["input"][i]["shape"][-1]) for i in range(num_encs)
# ]
# odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])


# parser = get_parser()
# args, _ = parser.parse_known_args(cmd_args)
# if args.dict is not None:
#     with open(args.dict, "rb") as f:
#         dictionary = f.readlines()
#     char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
#     char_list.insert(0, "<blank>")
#     char_list.append("<eos>")
#     if "maskctc" in args.model_module:
#         char_list.append("<mask>")
#     args.char_list = char_list
# set_deterministic_pytorch(args)

# model_module = args.model_module
# model_class = dynamic_import(model_module)
# model_class.add_arguments(parser)
# #print(args)
# model = model_class(idim_list[0],odim,args)
    
# E2E = model

# classifier = AccentClassifier(10)
# classifier.load_state_dict(torch.load("/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/classifier/snapshot.ep.10")['state_dict'])
# print(
#     " Total parameter of the model = "
#     + str(sum(p.numel() for p in classifier.parameters()))
# )
# model = DANN_ASR(E2E,classifier,args)
model_dir = "exp/mtl_lamda10_clf20ep/results/"
#model.load_state_dict(torch.load(model_dir+"model.loss.best"))
print(torch.load(model_dir+"model.loss.best"))
# print(
#     " Total parameter of the model = "
#     + str(sum(p.numel() for p in model.parameters()))
# )
