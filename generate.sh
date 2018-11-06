#!/usr/bin/env bash
#
# generate.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


export CUDA_VISIBLE_DEVICES=7

python generate.py \
    --vocab_path ./data/vocab.pkl \
    --checkpoint ./models/checkpoint.epoch-1.pth \
    --rnn_type GRU \
    --embedding_size 256 \
    --hidden_size 256 \
    --bidirectional \
    --num_layers 2 \
    --dropout 0.8 \
    --row 4 \
    --column 5 \
    --author 李白 \
    --title 友人 \
    --device cuda \
    --log_interval 100 \
    --temperature 1.0

/
