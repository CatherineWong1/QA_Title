# -*- encoding: utf-8 -*-
"""
Version: 1213
Update:
1. Add logger info for one epoch; Delete logger info for one batch data
2. Add torch save;
3. remove embedding initialize from model to train_iters();
4. Add load checkpoint
"""

import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import EncoderRnn, DecoderRnn, GreedySearchDecoder
import os
import json
import nltk
from nltk.corpus import stopwords
from data_loader import Dataset
import logging
from torch import optim
from pyrouge import Rouge155


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="Model")

PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_lengths = 10

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
logger.info("Device is {}".format(device))


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
    target_gate = target_gate.unsqueeze(1)
    loss_2 = F.binary_cross_entropy(vg_p, target_gate.float())
    loss = loss_1 + loss_2
    loss = loss.to(device)
    return loss


def train(input, lengths,target_gate, target_tensor, max_target_len, encoder_optimizer, decoder_optimizer,
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
    input = input.to(device)
    lengths = lengths.to(device)
    target_gate = target_gate.to(device)
    target_tensor = target_tensor.to(device)

    # Initial paramters
    loss = 0
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input, lengths)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:1]

    for timestep in range(max_target_len):
        predict_output, vg_p,decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # topk:return values and indicies
        _, top_ind = predict_output.topk(1)
        decoder_input = torch.LongTensor([[top_ind[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        vg_p = vg_p.to(device)
        # Calculate and accumulate loss
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
    logger.info("Initializing......")
    data = Dataset(args.data_path, args.batch_size)
    abstract_size = len(data.abstract_vocab)
    ordinary_size = len(data.ordinary_vocab)
    better_loss = 0
    logger.info("Initializing Encoder, Decoder and Optimizer")
    en_embedding = nn.Embedding(abstract_size, args.input_size)
    de_embedding = nn.Embedding(ordinary_size+10, args.dec_hidden_size)
    encoder = EncoderRnn(args.input_size, args.enc_hidden_size, 1, args.enc_drop, en_embedding)
    decoder = DecoderRnn(args.dec_hidden_size, 1, 10, ordinary_size, args.dec_drop, args.batch_size, de_embedding)
    # set to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)

    encoder.train()
    decoder.train()

    start_epoch = 1
    if args.load_model:
        checkpoint = torch.load(args.load_model)
        en_embedding.load_state_dict(checkpoint['en_embedding'])
        de_embedding.load_state_dict(checkpoint['de_embedding'])
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        start_epoch = checkpoint['epochs']

    # configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    logger.info("Training......")
    # train num epoch
    start_epoch += 1 if start_epoch != 1 else  start_epoch
    for i in range(start_epoch, args.epochs):
        logger.info("This is {} Epochs".format(i))
        iterations = 1
        epoch_loss = 0
        for batch in data.create_batch():
            batch_abstract = padding_batch_abstract(batch[0], batch[3])
            batch_title_ind, batch_title_type = padding_batch_title(batch[1], batch[2], batch[4])
            abstract_lengths = batch[3]
            title_lengths = batch[4]
            inputs = torch.tensor(batch_abstract).transpose(0,1)
            lengths = torch.tensor(abstract_lengths)
            target_gate = torch.tensor(batch_title_type).transpose(0,1)
            target_tensor = torch.tensor(batch_title_ind).transpose(0,1)
            targets = F.one_hot(target_tensor, num_classes=ordinary_size+10)
            loss = train(inputs, lengths, target_gate, targets, max(title_lengths), encoder_optimizer, decoder_optimizer,
                         encoder, decoder, args.batch_size)
            epoch_loss += loss
            iterations += 1
            if iterations % 50 == 0:
                logger.info("Loss avg of {} iterations is {}".format(iterations, epoch_loss/iterations))

        # Save Checkpoint per epoch
        model_name = "{}_epoch_model.tar".format(i)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        save_directory = os.path.join(args.model_path, model_name)
        torch.save({
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': epoch_loss/iterations,
            'en_embedding': en_embedding.state_dict(),
            'de_embedding': de_embedding.state_dict(),
            "epochs": i
        }, save_directory)

        logger.info("Epoch {} Loss is {}".format(i, epoch_loss/iterations))

    # Finish Training
    logger.info("Finish Training!")

    # 进入evaluate阶段
    logger.info("Start Evaluating")
    rouge_dir = "./rouge_dir"
    eval_iters(encoder, decoder, data, rouge_dir)


def evaluate(searcher, sentences, lengths, ordianry_vocab, topic_words):
    sentences = sentences.to(device)
    lengths = lengths.to(device)

    tokens, scores = searcher(sentences, lengths, MAX_lengths, SOS_token)
    
    # indexs -> words
    batch_outputs = [] 
    for i in range(tokens.size(0)):
        sent_inds = tokens[i]
        sum_vocab = ordianry_vocab + topic_words[i]
        sent = [sum_vocab[word_ind.item()] for word_ind in sent_inds]
        batch_outputs.append(" ".join(sent))

    return batch_outputs


def cal_rouge(rouge_dir):
    tmp_dir = "/home/huqian/anaconda3/envs/drqa_env/lib/python3.6/site-packages/pyrouge"
    r = Rouge155(rouge_dir=tmp_dir)
    r.model_dir = rouge_dir + "/reference"
    r.system_dir = rouge_dir + "/candidate"
    r.model_filename_pattern = 'ref.(\d+)_(\d+).txt'
    r.system_filename_pattern = 'cand.(\d+)_(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    result_dict = r.output_to_dict(rouge_results)

    return ">> Rouge - F(1/2//l): {:.2f}/{:.2f}/{:.2f}\n ROUGE- R(1/2//l): {:.2f}/{:.2f}/{:.2f} \n".format(
        result_dict["rouge_1_f_score"] * 100,
        result_dict["rouge_2_f_score"] * 100,
        result_dict["rouge_l_f_score"] * 100,
        result_dict["rouge_1_recall"] * 100,
        result_dict["rouge_2_recall"] * 100,
        result_dict["rouge_l_recall"] * 100
    )


def eval_iters(encoder, decoder, data, rouge_dir):
    """

    :param encoder:
    :param decoder:
    :return:
    """
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder, device)

    if not os.path.isdir(rouge_dir):
        os.mkdir(rouge_dir)
        os.mkdir(rouge_dir + "/candidate")
        os.mkdir(rouge_dir + "/reference")

    # load test
    batch_num = 1
    for test_batch in data.create_test():
        raw_title = test_batch[3]
        for i in range(len(raw_title)):
            title = raw_title[i]
            with open(rouge_dir + "/reference/ref.{}_{}.txt".format(batch_num, i+1),'w') as f:
                f.write(title)
      
        batch_abstract = padding_batch_abstract(test_batch[0], test_batch[2])
        inputs = torch.tensor(batch_abstract).transpose(0, 1)
        lengths = torch.tensor(test_batch[2])
        topic_words = test_batch[4]
        batch_outputs = evaluate(searcher, inputs, lengths, data.ordinary_vocab, topic_words)
        for j in range(len(batch_outputs)):
            output = batch_outputs[j]
            with open(rouge_dir + "/candidate/cand.{}_{}.txt".format(batch_num, j+1),'w') as f:
                f.write(output)
        batch_num += 1

    # 调用Rouge计算
    logger.info("Calculating Rouge Score")
    final_res = cal_rouge(rouge_dir)
    print(final_res)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/patent/')
    parser.add_argument('--input_size',type=int, default=300, help='word embedding dim')
    parser.add_argument('--enc_hidden_size', type=int, default=128)
    parser.add_argument('--dec_hidden_size',type=int, default=128)
    parser.add_argument('--enc_drop',type=float, default=0)
    parser.add_argument('--dec_drop',type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--decoder_learning_ratio", type=float, default=5.0)
    parser.add_argument("--model_path", type=str, default="./model/")
    parser.add_argument("--load_model", type=str, default="./model/1_epoch_model.tar")
    args = parser.parse_args()

    train_iter(args)

