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

from tqdm import tqdm

from modules.model import PoetGenModel
from misc.dataset import PoetDataset
from misc.vocab import Vocab


parser = argparse.ArgumentParser(description='Poet Generation.')

parser.add_argument('--data_path', type=str,
                    default='./data/poet_tang_simplified.txt', help='location of the data corpus')

parser.add_argument('--vocab_path', type=str,
                    default='./data/vocab.pkl', help='location of the vocab.')

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

parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')

parser.add_argument('--clip', type=float, default=50.0,
                    help='gradient clipping')

parser.add_argument('--epochs', type=int, default=15,
                    help='upper epoch limit')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--device', type=str,
                    help='use CUDA or CPU')

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--model_path', type=str,  default='./models/model.pth',
                    help='path to save the final model')

args = parser.parse_args()

device = torch.device(args.device)

torch.manual_seed(args.seed)

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

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train_epoch():
    model.train()
    start_time = time.time()
    max_iter = dataset.get_size('train')
    for epoch in range(1, args.epochs):
        total_loss = 0
        for iter in range(1, max_iter):
            row_tensor, column_tensor, author_tensor, \
                title_tensor, paragraphs_tensor, tags_tensor, _ = dataset.next_data(
                    'train')

            output = model(row_tensor,
                           column_tensor,
                           author_tensor,
                           title_tensor,
                           paragraphs_tensor[:-1, :],
                           tags_tensor)

            loss = 0

            optimizer.zero_grad()

            # loss
            loss = criterion(output.view(-1, vocab_size),
                             paragraphs_tensor[1:, :].view(-1))

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            total_loss += loss.item()

            if iter % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time

                print('| epoch {:3d} | {:5d}/{:5d} iter | lr {:02.4f} | ms/iter {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, iter,  max_iter, args.lr,
                          elapsed * 1000 / args.log_interval, avg_loss, math.exp(avg_loss)))
                total_loss = 0
                start_time = time.time()

        # test
        test()

        # save model of each epoch
        save_state = {
            'loss': avg_loss,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_path, 'checkpoint.epoch-%d.pth' % (epoch)))

        # eval 
        eval()


def test():
    model.eval()
    with torch.no_grad():
        datas = dataset.load_all('test')
        total_loss = 0
        for i, data in enumerate(tqdm(datas)):
            start_time = time.time()

            row_tensor, column_tensor, author_tensor, \
                title_tensor, paragraphs_tensor, tags_tensor, _ = data

            output = model(row_tensor,
                           column_tensor,
                           author_tensor,
                           title_tensor,
                           paragraphs_tensor[:-1, :],
                           tags_tensor)

            # loss
            loss = criterion(output.view(-1, vocab_size),
                             paragraphs_tensor[1:, :].view(-1))

            total_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                avg_loss = total_loss // args.log_interval
                print('time: {:5.2f}s |  loss {:5.2f} | '
                      'test ppl {:8.2f}'.format((time.time() - epoch_start_time),
                                                avg_loss, math.exp(avg_loss)))
                total_loss = 0
                start_time = time.time()


def eval():
    model.eval()
    with torch.no_grad():
        datas = dataset.load_all('eval')
        for data in tqdm(datas):
            row_tensor, column_tensor, author_tensor, \
                title_tensor, paragraphs_tensor, tags_tensor, paragraphs = data

            max_words = row_tensor.item() * column_tensor.item() * 2 + 20
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
            print('paragraphs: %s \n' % paragraphs)
            print('----------------------------')


def save_checkpoint(state, is_best, filename):
    '''
    Saving a model in pytorch.
    :param state: is a dict object, including epoch, optimizer, model etc.
    :param is_best: whether is the best model.
    :param filename: save filename.
    :return:
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best_%s.pth' % opt.model_type)


if __name__ == '__main__':
    train_epoch()
