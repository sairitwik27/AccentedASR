#!/usr/bin/env bash


# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4        # start from 0 if you need to start from data preparation
stop_stage=4
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
#resume=/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/exp/nisp_hin_pytorch_ind_finetune/results/snapshot.ep.5        # Resume the training from snapshot
# resume=/home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/exp/dat_lamda0.1_lr_1/results/snapshot.ep.10
resume=""
seed=1

# feature configuration
do_delta=false

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

# config files
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train_transfer.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
skip_lm_training=false  # for only using end-to-end ASR model without LM
use_wordlm=true         # false means to train/use a character LM
lm_vocabsize=65000      # effective only for word LMs
lm_resume=              # specify a snapshot file to resume LM training
lmtag=                 # tag for managing LMs

# decoding parameter
recog_model=model.loss.best   # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=3                 # the number of ASR models to be averaged
use_valbest_average=false    # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged

# exp tag
#tag="final" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set="train_NPTEL_IITM"
train_set="nisp_train"
train_dev=nisp_test
train_test=nisp_test
# recog_set="nisp_hin nisp_kan nisp_mal nisp_tam nisp_tel"
#recog_set=dev_IITM


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
# feat_dt_dir=dat_lamda0.1_nisp/val

ab=`nvidia-smi --query-gpu=index,memory.used --format=csv`
echo $ab
zero=`echo $ab | awk '{print $5}'`
one=`echo $ab | awk '{print $8}'`
gpu=0
if [ $zero -le  $one ] ;then
gpu=0
else
gpu=1
fi
echo "using gpu ${gpu}"

dict=data/lang_1char/train_NPTEL_IITM_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
expname="dat_lamda0.1_lr_1_nisp_ft"
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

   CUDA_VISIBLE_DEVICES=${gpu} ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        /home/ritwikkvs/ritwik/espnet/egs/wsj/ind_asr/local/dann_asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir "" \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json dump/nisp_train/deltafalse/data_with_labels.json \
        --valid-json dump/nisp_test/deltafalse/data_with_labels.json
fi
