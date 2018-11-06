#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
train
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim


from modules.model import PoetGenModel
from misc.dataset import PoetDataset
from misc.vocab import Vocab


parser = argparse.ArgumentParser(description='Poet Generation.')

parser.add_argument('--vocab_path', type=str,
                    default='./data/vocab.pkl', help='location of the vocab.')

parser.add_argument('--checkpoint', type=str,  default='./models/model.pth',
                    help='path to save the final model')

parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

parser.add_argument('--embedding_size', type=int, default=256,
                    help='size of word embeddings')

parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units per layer')

parser.add_argument('--bidirectional', action='store_true',
                    help='is bidirectional rnn.')

parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--row', type=int,  default=4,
                    help='row of poet.')

parser.add_argument('--column', type=int,  default=5,
                    help='column of poet.')

parser.add_argument('--author', type=str,
                    help='author of poet.')

parser.add_argument('--title', type=str,
                    help='title of poet.')

parser.add_argument('--device', type=str,
                    help='use CUDA or CPU')

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity.')

args = parser.parse_args()

device = torch.device(args.device)

vocab = Vocab()
vocab.load(args.vocab_path)
vocab_size = vocab.size
print('vocab_size: %d ' % vocab_size)


split = {
    'train': 0.8,
    'test': 0.15,
    'eval': 0.05
}
dataset = PoetDataset(args.data_path,
                      vocab,
                      split)

model = PoetGenModel(vocab_size,
                     dataset.max_row,
                     dataset.max_cloumn,
                     dataset.author_num,
                     args.embedding_size,
                     args.hidden_size,
                     args.num_layers,
                     args.bidirectional,
                     args.rnn_type,
                     args.dropout,
                     device)

model.load_state_dict(checkpoint['state_dict'])


def generate():
    model.eval()
    with torch.no_grad():

        row_tensor = torch.tensor([args.row], dtype=torch.long, device=device)
        column_tensor = torch.tensor(
            [args.column], dtype=torch.long, device=device)
        author_tensor = torch.tensor([dataset.author_index(
            args.author)], dtype=torch.long, device=device)

        title_ids = vocab.words_to_id([char for char in args.title])
        title_tensor = torch.tensor(
            title_ids, dtype=torch.long, device=self.device).view(-1, 1)

        tags_tensor = None
        max_words = args.row * args.column * 2 + 20
        input_tensor = torch.ones(
            (1, 1), dtype=torch.long, device=device) * vocab.sos_id
        poet = model.generate(row_tensor,
                              column_tensor,
                              author_tensor,
                              title_tensor,
                              tags_tensor,
                              input_tensor,
                              max_words)

        poet = [vocab.ids_to_word(poet)]
        print('poet: %s \n' % poet)


def load_checkpoint(state, is_best, filename):
    checkpoint = torch.load(filename)
    return checkpoint


if __name__ == '__main__':
    train_epoch()
