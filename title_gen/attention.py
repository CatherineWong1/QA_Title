# -*- encoding:utf-8 -*-
"""
Luong Attention和Bahdanau的Attention所不同的是:
1.
Luong Attention是一种global attention，考虑的是Encoder的all hidden states
2.
计算权重是使用的当前Decoder的hidden state
Bahdanau使用的是previous timestep的hidden state

这种Attention提供三种score function
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Attn(nn.Module):
    """
    Input:
    method: 选择score的计算方式
    hidden_size：decoder units

    Output:
    output: softmax normalized weigths tensor
    shape(batch_size,1, max_length)
    """
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot ','general','concat']:
            raise ValueError(self.method ,"is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size *2, hidden_size)
            # nn.Parameter make a tensor be a module parameter
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # calculate the attention weights based on method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # the softmax normalized probability socres
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)

        return attn_weights


class Attn_bahdanau(nn.Module):
    """
    Bahdanau Attention
    math:
       x = (input, output) 即（h1,h2,....hn)
       attn = exp(x_i) / sum_j exp(x_j)
       output = tanh(w * (attn*input) + b*output)
    Inputs:
        output: (batch, output_len, dimensions): containing output features from the decoder
        input:(batch, input_len, dimensions): containing features of encoded input sequence
    Outputs:
        context:(batch, output_len, dimensions): context
        attn: (batch, output_len, input_len): attention weights
    """
    def __init__(self, dim):
        super(Attn_bahdanau, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, output, inputs):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_len = inputs.size(1)

        # (batch,out_len,dim) * (batch,input_len, dim) ---> (batch,out_len,in_len)
        attn = torch.bmm(output, inputs.transpose(1,2))

        attn = F.softmax(attn.view(-1, input_len),dim=1).view(batch_size, -1, input_len)

        context = torch.bmm(attn, inputs)
        return context, attn


class AttenDecoder(nn.Module):
    """
    Create Decoder used Attention
    """
    def __init__(self,attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttenDecoder, self).__init__()

        # keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):

        # one timestep function, similar with HTD decoder_fn
        # get embedding of current input word
        embedded = self.embedding(input_step)
        embedding = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU
        # rnn_output (1, batch_size, hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # calculate normal attention weights from the current GRU outputs
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # calculate topic attention weights from current GPU

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # attn_weights (batch_size,1, max_length), encoder_outputs(max_length, batch_size，hidden_size)
        # context shape(batch_size,1, hidden_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # Concatenate weighted context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context),1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden



