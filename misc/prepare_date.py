#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
1. load json datas
2. traditional chinese to simplified chinese
3. stats (len, author, type)
4. build vocab
5. save
"""

import os
import re
import sys
import json
import argparse
import subprocess
from tqdm import tqdm

import pickle
from vocab import Vocab


parser = argparse.ArgumentParser(description='Poet Generation.')

# args
parser.add_argument('--data',
                    type=str,
                    default='../data/tang/',
                    help='location of the data corpus.')

parser.add_argument('--vocab_path',
                    type=str,
                    default='../data/vocab.pkl',
                    help='location of the vocab.')

parser.add_argument('--save_path_traditional',
                    type=str,
                    default='../data/poet_tang_traditional.txt',
                    help='location of saving transformed data.')

parser.add_argument('--save_path_simplified',
                    type=str,
                    default='../data/poet_tang_simplified.txt',
                    help='location of saving transformed data.')

parser.add_argument('--min_count',
                   type=int,
                   default=2,
                   help="Ignores all words with total frequency lower than this.")

args = parser.parse_args()


save_file = open(args.save_path_traditional, 'w', encoding='utf-8')
for filename in tqdm(os.listdir(args.data)):
    if not filename.startswith('poet'):
        continue

    file_path = os.path.join(args.data, filename)

    poet_list = json.load(open(file_path, 'r', encoding='utf-8'))

    for poet in poet_list:
        author = poet['author']
        paragraphs = poet['paragraphs']
        new_paragraphs = []
        for paragraph in paragraphs:
            index = paragraph.find('{')
            if index != -1:
                first_word = paragraph[index + 1]
                paragraph = re.sub(r'\{.+}', first_word, paragraph)

            paragraph = re.sub(r'（.+）', '', paragraph)

            paragraph.replace('[', '')
            paragraph.replace(']', '')

            if paragraph.find('（') != -1 or paragraph.find('{') != -1 or paragraph.find('《') != -1 or paragraph.find('[') != -1:
                continue

            if len(paragraph) <= 2:
                continue

            new_paragraphs.append(paragraph)

        if len(new_paragraphs) == 0:
            continue
        row_num = len(new_paragraphs)
        column_num = (len(new_paragraphs[0]) - 2) // 2

        paragraphs = ''.join(new_paragraphs)
        #  strains = ''.join(poet['strains'])
        title = poet['title']
        tags = ' '.join(poet.get('tags', []))

        save_file.write('%d\t%d\t%s\t%s\t%s\t%s\n' % (row_num, column_num, author, title, paragraphs, tags))

save_file.close()

# opencc traditional -> simplified
# opencc -i comments.txt -o comments.txt -c t2s.json
cmd = [
    'opencc',
    '-i', '%s' % args.save_path_traditional,
    '-o', '%s' % args.save_path_simplified,
    '-c', 't2s.json'
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
_, error = process.communicate()

#
def save_distribution(distribution_list, name):
    """
    distribution: [(i, j), (i, j), ...]
    """
    with open(name + '.distribution.txt', 'w', encoding="utf-8") as f:
        for i, j in distribution_list:
            f.write('%s\t%s\n' % (str(i), str(j)))
#
vocab = Vocab()
author_dict = {}
title_dict = {}
len_dict = {} # d-d (row, column)
with open(args.save_path_simplified, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        parts = line.rstrip().split('\t')
        row_num = parts[0]
        column_num = parts[1]
        author = parts[2]
        title = parts[3]
        paragraphs = parts[4]
        if len(parts) >= 6:
            tags = parts[5]

        vocab.add_words([char for char in title])
        vocab.add_words([char for char in paragraphs])

        author_dict[author] = author_dict.get(author, 0) + 1
        title_dict[title] = title_dict.get(title, 0) + 1
        len_dict[str(row_num) + '-' + str(column_num)] = len_dict.get(str(row_num) + '-' + str(column_num), 0) + 1

    sorted_list = vocab.build_vocab(args.min_count)
    vocab.save(args.vocab_path)

save_distribution(sorted_list, 'vocab_freq')
save_distribution(sorted(len_dict.items(), key=lambda item: item[1], reverse=True), 'len_freq')
save_distribution(sorted(title_dict.items(), key=lambda item: item[1], reverse=True), 'title_freq')
save_distribution(sorted(author_dict.items(), key=lambda item: item[1], reverse=True), 'author_freq')


print('----prepare finished---------')

