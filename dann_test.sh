#!/bin/bash

# Copyright 2018 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=9       # start from 0 if you need to start from data preparation
stop_stage=9
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode_default.yaml

# rnnlm related
use_wordlm=false     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="dat_lamda0.1_lr_1_nisp" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



train_set=train_NPTEL_IITM
train_dev=dev_NPTEL
train_test=dev_IITM
recog_set="nisp_test"
#recog_set=dev_IITM

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/train_NPTEL_IITM_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt



# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
#lmexpname=train_rnnlm_${backend}_${lmtag}

#Modification: you dont need to install rnn-lm again and again or copy it to your folder
#if you have train the folder once, its okay just mention the path of the folder that is wwhere they have install the 
#rnnlm
#lmexpdir=/home/rohitk/Workspace/E2E/espnet/egs/reverb/asr1_test/exp/${lmexpname}
#mkdir -p ${lmexpdir}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}



if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi

lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}

expdir=exp/${expname}
mkdir -p ${expdir}

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
gpu=0

expdir=exp/dat_lamda0.1_lr_1_nisp_new
model_expdir=exp/dat_lamda0.1_lr_1_nisp
JOBS=30
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 7: Decoding"
    nj=$JOBS
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if [ ${use_wordlm} = true ]; then
            decode_dir=${decode_dir}_wordrnnlm_${lmtag}
        else
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        mkdir -p ${expdir}/${decode_dir}/log

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_with_labels.json

        for JOBS in `seq 1 $nj`; do
        qsub -q med.q -l hostname=compute-0-[2-3] -V -cwd -e ${expdir}/${decode_dir}/log/decode.${JOBS}.log -o ${expdir}/${decode_dir}/log/decode.${JOBS}.log -S /bin/bash asr_recog_code.sh ${feat_recog_dir}/split${nj}utt/data_with_labels.${JOBS}.json ${expdir}/${decode_dir}/data_with_labels.${JOBS}.json ${model_expdir}
        sleep 3s
        done

    )
    done

    echo "Finished"
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Decoding Results"
    nj=$JOBS

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        if [ ${use_wordlm} = true ]; then
            decode_dir=${decode_dir}_wordrnnlm_${lmtag}
        else
            decode_dir=${decode_dir}_rnnlm_${lmtag}
        fi
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        #mkdir -p ${expdir}/${decode_dir}_with_jobs/log
        # split data
        #splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    )
    done
fi