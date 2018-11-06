#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from modules import embeddings
from modules.rnn_encoder import RNNEncoder
from modules.decoder import StdRNNDecoder


"""
1. row, column embedding
2. title encoder
3. concat -> linear
4. decoder
"""

class PoetGenModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_row,
                 max_cloumn,
                 author_num,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 rnn_type,
                 dropout,
                 device):
        super(PoetGenModel, self).__init__()

        self.row_embedding = nn.Embedding(
            max_row,
            embedding_size
        )

        self.column_embedding = nn.Embedding(
            max_cloumn,
            embedding_size
        )

        self.author_embedding = nn.Embedding(
            author_num,
            embedding_size
        )
        # title encoder
        encoder_embedding = embeddings.Embeddings(
            embedding_size,
            vocab_size,
            word_padding_idx=0,
            dropout=dropout
        )
        self.encoder = RNNEncoder(
            rnn_type,
            bidirectional,
            num_layers,
            hidden_size,
            dropout,
            encoder_embedding
        )

        # decoder

        self.decoder = StdRNNDecoder(
            rnn_type,
            bidirectional,
            num_layers,
            hidden_size,
            dropout=dropout,
            embeddings=encoder_embedding
        )

        self.output_linear = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)



    def forward(self,
                row_tensor,
                column_tensor,
                author_tensor,
                title_tensor,
                paragraphs_tensor,
                tags_tensor):
        """
        row_tensor: [1, 1]
        column_tensor: [1, 1]
        author_tensor: [1, 1]
        title_tensor: [len, 1]
        paragraphs_tensor: [len, 1]
        tags_tensor: [len, 1]
        """

        row_embedded = self.row_embedding(row_tensor) # [1, 1, embedding_size]
        column_embedded = self.column_embedding(column_tensor) #[1, 1, embedding_size]
        author_embedded = self.author_embedding(author_tensor) #[1, 1, embedding_size]

        # encoder
        hidden_final, memory_bank, _ = self.encoder(title_tensor, row_embedded, column_embedded, author_embedded)

        # hidden_final: [num_layers * bidirection_num, 1, hidden_size // 2]
        self.decoder.init_state(hidden_final)

        output, attn = self.decoder(paragraphs_tensor, memory_bank)
        # output: [len, 1, hidden_size]

        output = self.output_linear(output) # [len, 1, vocab_size]

        output = self.log_softmax(output) # [len, 1, vocab_size]

        return output


    def generate(self,
                 row_tensor,
                 column_tensor,
                 author_tensor,
                 title_tensor,
                 tags_tensor,
                 input_tensor,
                 max_words):
        """
        input_tensor: [1, 1] SOS_id
        """

        row_embedded = self.row_embedding(row_tensor) # [1, 1, embedding_size]
        column_embedded = self.column_embedding(column_tensor) #[1, 1, embedding_size]
        author_embedded = self.author_embedding(author_tensor) #[1, 1, embedding_size]

        # encoder
        hidden_final, memory_bank, _ = self.encoder(title_tensor, row_embedded, column_embedded, author_embedded)

        # hidden_final: [num_layers * bidirection_num, 1, hidden_size // 2]
        self.decoder.init_state(hidden_final)

        poet = []
        for i in range(max_words):
            output, attn = self.decoder(input_tensor, memory_bank)
            output = self.output_linear(output) # [1, 1, vocab_size]
            output = self.log_softmax(output) # [1, 1, vocab_size]

            input_tensor = output.argmax(dim=2).detach().view(-1, 1)
            poet.append(input_tensor.item())

        return poet












