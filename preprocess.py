# -*- encoding:utf-8 -*_
"""
1. 训练LDA模型或者Topic Rank模型
2. 将所有数据送入1中train好的模型

Version1:
研究了lda模型，发现并不符合我们的场景

Version2:
利用topic rank生成关键词

"""
import os
import random
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases
from gensim.models import LdaModel


def shuffle_dataset(data_path):
    """
    spilt raw data into training dataset and testing dataset
    :param data_path: raw data path
    :return: shuffle file and rename
    """
    file_list = os.listdir(data_path)
    file_nums = len(file_list)
    random.shuffle(file_list)
    for i in range(file_nums):
        old_name = data_path + "/" + file_list[i]
        if i < 50:
            new_name = data_path + "/train/" + "train.{}.txt".format(i+1)
        else:
            new_name = data_path + "/test/" + "train.{}.txt".format(i-49)
        os.rename(old_name, new_name)


def generate_vocab(train_path):
    """
    generate two vocabularies for train dataset: abstract vocabulary and title vocabulary
    :param data_path:
    :return:
    """
    file_list = os.listdir(train_path)
    stop_list = stopwords.words('english')
    abstract_vocab = {}
    title_vocab = {}
    for file in file_list:
        filename = train_path + "/" + file
        with open(filename) as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                    abstract = data['abstract'].lower()
                    abstract = re.sub(r'[^\x00-\x7F]','', abstract)
                    abstract = abstract.split(" ")
                    title = data['patent_name'].lower().split(" ")
                    for word in abstract:
                        if word not in stop_list:
                            if not word in abstract_vocab:
                                abstract_vocab[word] = 0
                            abstract_vocab[word] += 1

                    for word in title:
                        if word not in stop_list:
                            if not word in title_vocab:
                                title_vocab[word] = 0
                            title_vocab[word] += 1
                except Exception as e:
                    continue

    return abstract_vocab, title_vocab


def process_lda_data(abstract):
    """
    process abstract for lda
    :param abstract: abstract list
    :return: formatted data for lda model
    """
    docs = []
    # remove numeric tokens and single character
    tokenizer = RegexpTokenizer(r'\w+')
    for item in abstract:
        word_list = tokenizer.tokenize(item)
        word_list = [word for word in word_list if len(word) > 1]
        docs.append(word_list)

    # compute bigrams, add phrase like machine_learning
    # I used very small corpus to train this model and used the same sentences to inference. No working
    bigram = Phrases(docs, min_count=5)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                docs[idx].append(token)

    return docs


def train_lda(corpus_path):
    """
    train lda model
    :param corpus_path: data path
    :return:
    """
    abstract_list = []
    # load lda corpus
    file_list = os.listdir(corpus_path)
    file_list = [corpus_path+"/"+file for file in file_list]
    for file in file_list:
        with open(file) as f:
            for line in f.readlines():
                data = json.loads(line)
                abstract = data['abstract'].lower()
                abstract = re.sub(r'[^\x00-\x7F]','', abstract)
                abstract_list.append(abstract)

    # process lda data
    docs = process_lda_data(abstract)
    dictionary = Dictionary(docs)

    # bag-of-words
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print("Number of unique tokens: %d" % len(dictionary))

    # set training parameters
    num_topic = 3
    chunk_size = 1000
    passes = 20
    iterations = 2000

    temp = dictionary[0]
    id2word = dictionary.id2token
    model = LdaModel(corpus=corpus, num_topics=num_topic, id2word=id2word, chunksize=chunk_size,passes=passes,
                     alpha='auto', eta='auto',eval_every=None,iterations=iterations)

    top_topics = model.top_topics(corpus, topn=5)


    #avg_topics_coherence = sum([t[1] for t in top_topics]) / num_topic
    from pprint import pprint
    pprint(top_topics)

    # remain to generate key word vocabulary
