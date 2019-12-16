# -*- encoding: utf-8 -*-

import argparse
import torch
import torch.nn.functional as F
from model import EncoderRnn, DecoderRnn
import os
import json
import nltk
from nltk.corpus import stopwords
from data_loader import Dataset
import logging
from torch import optim


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="Model")

PAD_token = 0
SOS_token = 1
EOS_token = 2

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
logger.info("The device is {}".format(device))

def padding_batch_abstract(batch_abstract, lengths):
    max_length = max(lengths)
    batch_abstract = [x + [0]*(max_length-len(x)) for x in batch_abstract]
    return batch_abstract


def padding_batch_title(batch_title_ind, batch_title_type, lenghts):
    max_length = max(lenghts)
    batch_title_ind = [x + [0]*(max_length-len(x)) for x in batch_title_ind]
    batch_title_type = [x + [0]*(max_length-len(x)) for x in batch_title_type]
    return batch_title_ind, batch_title_type


def criterion(predict_output, target_output, vg_p, target_gate):
    """
    Calculate loss for every timestep
    :param predict_output: (batch_size, vocab_size)
    :param target_output: (batch_size,vocab_size)
    :param vg_p: (batch_size,2)
    :param target_gate: (batch_size,2)
    :return:
    """
    loss_1 = F.binary_cross_entropy(predict_output, target_output.float())
    loss_2 = F.binary_cross_entropy(vg_p, target_gate.float())
    return loss_1 + loss_2


def train(inputs, lengths,target_gate, target_tensor, max_target_len, encoder_optimizer, decoder_optimizer,
          encoder, decoder, batch_size):
    """
    train for every time step
    :param input:
    :param lengths: batch input have different lengths
    :param target_gate:
    :param target_tensor:
    :param max_target_len:
    :return:
    """
    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    #inputs = inputs.to(device)
    #lengths = lengths.to(device)
    #target_gate = target_gate.to(device)
    #target_tensor = target_tensor.to(device)

    # Initial paramters
    loss = 0
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(inputs, lengths)
    
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    #decoder_input = decoder_input.to(device)
  
    decoder_hidden = encoder_hidden[:1]
    
    for timestep in range(max_target_len):
        predict_output, vg_p,decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # topk:return values and indicies
        _, top_ind = predict_output.topk(1)
        decoder_input = torch.LongTensor([[top_ind[i][0] for i in range(batch_size)]])
        #decoder_input = decoder_input.to(device) 

        # Calculate and accumulate loss
        # 如果timestep不一致怎么办？
        loss += criterion(predict_output, target_tensor[timestep], vg_p, target_gate[timestep])
        #print("The loss of timestep {} is {}".format(timestep, loss))
        #if decoder_input.item() == EOS_token:
        #    break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss / max_target_len


def train_iter(args):
    # get batch data, vocab size
    logger.info("Initializing Dataset and Vocab......")
    data = Dataset(args.data_path, args.batch_size , 'train')
    abstract_size = len(data.abstract_vocab)
    ordinary_size = len(data.ordinary_vocab)
    total_loss = 0
    logger.info("Initializing Encoder,Decoder and Optimizer......")
    encoder = EncoderRnn(args.input_size, args.enc_hidden_size, 1, abstract_size, args.enc_drop)
    decoder = DecoderRnn(args.dec_hidden_size, 1, 10, ordinary_size, args.dec_drop, args.batch_size)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate*args.decoder_learning_ratio)
    logger.info("Training......")
    # train num epoch
    for i in range(1, args.epochs):
        iterations = 1
        for batch in data.create_batch():
            batch_abstract = padding_batch_abstract(batch[0], batch[3])
            batch_title_ind, batch_title_type = padding_batch_title(batch[1], batch[2], batch[4])
            abstract_lengths = batch[3]
            title_lengths = batch[4]
            inputs = torch.tensor(batch_abstract).transpose(0,1)
            lengths = torch.tensor(abstract_lengths)
            target_gate = torch.tensor(batch_title_type).transpose(0,1)
            target_tensor = torch.tensor(batch_title_ind).transpose(0,1)
            targets = F.one_hot(target_tensor,num_classes=ordinary_size+10)
            loss = train(inputs, lengths, target_gate, targets, max(title_lengths),encoder_optimizer, decoder_optimizer,
                         encoder, decoder, args.batch_size)
            total_loss += loss
            iterations += 1
            if iterations % 10 == 0:
                logger.info("Loss avg of {} iterations is {}".format(iterations,total_loss/iterations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='./data/patent/')
    parser.add_argument('--input_size',type=int, default=300, help='word embedding dim')
    parser.add_argument('--enc_hidden_size', type=int, default=128)
    parser.add_argument('--dec_hidden_size',type=int, default=128)
    parser.add_argument('--enc_drop',type=float, default=0)
    parser.add_argument('-dec_drop',type=float, default=0)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--decoder_learning_ratio", type=float, default=5.0)
    args = parser.parse_args()
    if args.mode == 'train':
        train_iter(args)



