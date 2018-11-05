#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


export CUDA_VISIBLE_DEVICES=7

python train.py \
    --data_path ./data/poet_tang_simplified.txt \
    --vocab_path ./data/vocab.pkl \
    --rnn_type GRU \
    --embedding_size 256 \
    --hidden_size 256 \
    --bidirectional \
    --num_layers 2 \
    --lr 0.001 \
    --clip 5.0 \
    --dropout 0.8 \
    --tied \
    --epochs 10 \
    --seed 7 \
    --device cuda \
    --log_interval 100 \
    --model_path ./models

/
