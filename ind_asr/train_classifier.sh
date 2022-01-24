#!/bin/sh
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# python3 local/transformer_embeddings.py
python3 local/clf_tr.py
# python3 local/accent_embeddings.py