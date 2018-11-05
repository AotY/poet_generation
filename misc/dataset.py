#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
dataset
"""

import os
import io
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


class PoetDataset:
    def __init__(self, data_path, vocab, split=None, transform=None, device=None):
        """
        data_path: str
        transform: None
        split: dict: {'train': 0.8, 'test': 0.15, 'eval': 0.05}
        """
        self.data_path = data_path
        self.vocab = vocab
        self.transform = transform
        self.split = split
        self.device = device

        self.max_row = 0
        self.max_cloumn = 0

        self._authors = set()

        self._load_data()

    def _load_data(self):
        print('load data...')
        self._data = {}
        _datas = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.rstrip().split('\t')
                row_num = int(parts[0])
                column_num = int(parts[1])
                author = parts[2]
                title = parts[3]
                paragraphs = parts[4]
                tags = None
                if len(parts) >= 6:
                    tags = parts[5]

                # for simple
                if row_num >= 10 and column_num >= 10:
                    continue

                _datas.append((row_num, column_num, author, title, paragraphs, tags))
                self.max_row = max(row_num, self.max_row)
                self.max_cloumn = max(column_num, self.max_cloumn)
                self._authors.add(author)

        np.random.shuffle(_datas)
        self._authors = list(self._authors)
        self.author_num = len(self._authors)

        self._size_dict = {
            'train': int(len(_datas) * (1. - self.split['test'] - self.split['eval'])),
            'test': int(len(_datas) * self.split['test']),
        }
        self._size_dict['eval'] = len(_datas) - self._size_dict['train'] - self._size_dict['test']

        self._data_dict = {
            'train': _datas[0: self._size_dict['train']],
            'test': _datas[self._size_dict['train']: (self._size_dict['train'] + self._size_dict['test'])],
            'eval': _datas[self._size_dict['train'] + self._size_dict['test']: ]
        }

        self._indicator_dict = {
            'train': 0,
            'eval': 0,
            'test': 0
        }

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def get_size(self, task):
        return self._size_dict[task]

    def load_all(self, task):
        return self._data_dict[task]

    def next_data(self, task):
        self._indicator_dict[task] += 1
        if self._indicator_dict[task] >= self._size_dict[task]:
            self.reset_data(task)

        data = self._data_dict[task][self._indicator_dict[task]]
        row_num, column_num, author, title, paragraphs, tags = data

        # to tensor
        row_tensor = torch.tensor(row_num, dtype=torch.long, device=self.device).view(-1, 1)
        column_tensor = torch.tensor(column_num, dtype=torch.long, device=self.device).view(-1, 1)

        author_tensor = torch.tensor(self._authors.index(author), dtype=torch.long, device=self.device).view(-1, 1)

        title_ids = self.vocab.words_to_id([char for char in title])
        title_tensor = torch.tensor(title_ids, dtype=torch.long, device=self.device).view(-1, 1)

        paragraphs_ids = self.vocab.words_to_id([char for char in paragraphs])
        paragraphs_ids.append(self.vocab.eos_id)
        paragraphs_ids.insert(0, self.vocab.sos_id)
        paragraphs_tensor = torch.tensor(paragraphs_ids, dtype=torch.long, device=self.device).view(-1, 1)

        tags_tensor = None
        if tags is not None:
            tags_ids = self.vocab.words_to_id([char for char in tags])
            tags_tensor = torch.tensor(tags_ids, dtype=torch.long, device=self.device).view(-1, 1)

        return row_tensor, column_tensor, author_tensor, \
            title_tensor, paragraphs_tensor, tags_tensor

    def next_batch(self, task):
        pass



