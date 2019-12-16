# -*- encoding:utf-8 -*-
"""
本模块实现question的生成
Encoder 1层 BiGRU
Decoder 1层 GRU + normal attention + topic attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from attention import Attn_bahdanau


# Create Encoder
class EncoderRnn(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers, abstract_vocab_size,dropout):
        super(EncoderRnn, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.abstract_vocab_size = abstract_vocab_size
        self.embedding_dim = 300
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers ==1 else dropout),
                          bidirectional=True)

        self.embedding = nn.Embedding(self.abstract_vocab_size, self.input_size)

    def forward(self, input_index, input_lengths,hidden=None):
        """
        Encoder function
        :param input_index: shape(max_length,batch_size)
        :param input_lengths: list of sequence length,shape(batch_size)
        :param hidden: hidden_state
        :return: outputs, output features shape(max_length,batch_size, hidden_size);
                  hidden:last timestep hidden state
        """
        # packed input sequence
        embedded = self.embedding(input_index)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)

        # unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # sum bidirectional GRU outputs
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        return outputs, hidden


# Create Decoder
class DecoderRnn(nn.Module):
    def __init__(self, hidden_size,  n_layers, topic_size, ordinary_size, dropout, batch_size):
        super(DecoderRnn, self).__init__()
        # Parameters
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.topic_size = topic_size
        self.ordinary_size = ordinary_size

        # embedding init, vocab_size = topic
        vocab_size = topic_size + ordinary_size
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Attention and GRU
        self.attn_model = Attn_bahdanau(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        # Vocabulary gate
        self.vg_linear = nn.Linear(hidden_size, 1, bias=False)

        # generate function
        self.gt = nn.Linear(hidden_size,topic_size, bias=False)
        self.go = nn.Linear(hidden_size, ordinary_size, bias=False)

    def cal_vog(self, hidden, last_output, context):
        self.b_vog = nn.Parameter(torch.randn(self.batch_size, 1))
        self.in_vog = self.vg_linear(hidden) + self.vg_linear(last_output) + self.vg_linear(context) + self.b_vog
        self.p_vog = torch.sigmoid(self.in_vog)

        return self.p_vog

    def cal_prop(self, hidden, last_output):
        self.b_topic = nn.Parameter(torch.randn(self.batch_size, self.topic_size))
        self.in_topic = self.gt(hidden) + self.gt(last_output) + self.b_topic
        self.p_topic = F.softmax(input=self.in_topic, dim=1)

        self.b_ordinary = nn.Parameter(torch.randn(self.batch_size, self.ordinary_size))
        self.in_ordinary = self.go(hidden) + self.go(last_output) + self.b_ordinary
        self.p_ordinary = F.softmax(input=self.in_ordinary, dim=1)

        return self.p_topic, self.p_ordinary

    def forward(self, input_step, last_hidden, encoder_outputs):
        # get embedding of current input word
        # embedded:(1,batch,embedding_dim)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # GRU input: last hidden and last output
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights to encoder outputs to get new "weighted sum" context vector
        context, attn_weights = self.attn_model(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1))

        # decoder final output
        # rnn_output(1,batch,hidden_size),context(batch, 1,hidden_size) -- > (batch,hidden_size)
        rnn_output = rnn_output.squeeze(0)
        embedded = embedded.squeeze(0)
        context = context.transpose(0,1).squeeze(0)
        # (batch,2*hidden_size)
        concat_input = torch.cat((rnn_output, context), 1)

        vg_p = self.cal_vog(rnn_output, embedded, context)
        p_topic, p_ordinary = self.cal_prop(rnn_output, embedded)
        # (batch_size, ordinary_vocab+topic_vocab)
        final_output = torch.cat(((1-vg_p)*p_ordinary, vg_p*p_topic),1)

        return final_output, vg_p, hidden
