# -*- encoding:utf-8 -*-
"""
"""
import json
import logging
import glob2
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()


_Symbol_Vocab = ['PAD', 'SOS', 'EOS']

# setting logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name="Data")


def load_dataset(files):
    logger.info("Loading Data")

    def _lazy_dataset_loader(json_file):
        with open(json_file) as f:
            dataset = json.load(f)
            logger.info('Loading dataset from %s, number of examples: %d' % (json_file, len(dataset)))
            return dataset

    random.shuffle(files)
    for file in files:
        yield _lazy_dataset_loader(file)


class Dataset(object):
    def __init__(self, data_path, batch_size):
        self.files = sorted(glob2.glob(data_path + 'train' + '.[0-9]*.json'))
        self.test_files = sorted(glob2.glob(data_path + 'test' + '.[0-9]*.json'))
        self.abstract_vocab = []
        self.ordinary_vocab = []
        self.generate_abstract()
        self.generate_ordinary()
        self.batch_size = batch_size
        self.cur_iter = load_dataset(self.files)
        self.test_iter = load_dataset(self.test_files)

    def generate_abstract(self):
        """
        generate abstract vocab for encoder phase
        sort vocab with frequency
        :param data_path: train abstract data and test abstract data
        :param vocab_path: abstract vocab path
        :return:
        """
        logger.info("Generate Abstract Vocab List, according to the alphabet")
        for file in self.files:
            with open(file) as f:
                data_list = json.load(f)
                for data in data_list:
                    word_list = nltk.word_tokenize(data['abstract'])
                    for word in word_list:
                        if word not in self.abstract_vocab:
                            self.abstract_vocab.append(word)

        for file in self.test_files:
            with open(file) as f:
                data_list = json.load(f)
                for data in data_list:
                    word_list = nltk.word_tokenize(data['abstract'])
                    for word in word_list:
                        if word not in self.abstract_vocab:
                            self.abstract_vocab.append(word)

        self.abstract_vocab = _Symbol_Vocab + sorted(self.abstract_vocab)
        logger.info("Finished generating abstract vocab")

    def generate_ordinary(self):
        """
        Two function:
        1. get ordinary vocab
        2. get word type
        :param file_list:
        :param ordinary_vocab:
        :return:
        """
        logger.info("Generate Ordinary Vocab, according to the alphabet")
        for file in self.files:
            with open(file) as f:
                data_list = json.load(f)
                for data in data_list:
                    title_words = nltk.word_tokenize(data['patent_name'])
                    i = 0
                    while i < len(title_words) -1:
                        if title_words[i] not in data['topic']:
                            phrase = " ".join([title_words[i], title_words[i + 1]])
                            if phrase in data['topic']:
                                i += 2
                                continue
                            else:
                                if title_words[i] not in self.ordinary_vocab:
                                    self.ordinary_vocab.append(title_words[i])
                                if i + 1 == len(title_words) - 1:
                                    self.ordinary_vocab.append(title_words[i + 1])
                        i += 1

        self.ordinary_vocab = _Symbol_Vocab + sorted(self.ordinary_vocab)
        logger.info("Finished generating ordinary vocab")

    def generate_ind(self, sample):
        """
          generate tensor for each sample
          :param sample:
          :return: abstract_index, title_type, title_index
          """
        abstract_words = nltk.word_tokenize(sample['abstract'])
        abstract_ind = []
        for word in abstract_words:
            abstract_ind.append(self.abstract_vocab.index(word))
        title_words = nltk.word_tokenize(sample['patent_name'])
        title_ind = []
        title_type = []
        i = 0
        sum_title_vocab = self.ordinary_vocab + sample['topic']
        while i < len(title_words) - 1:
            if title_words[i] in sample['topic']:
                title_type.append(1)
                title_ind.append(sum_title_vocab.index(title_words[i]))

            if title_words[i] not in sample['topic']:
                phrase = " ".join([title_words[i], title_words[i + 1]])
                if phrase in sample['topic']:
                    title_type.append(1)
                    title_ind.append(sum_title_vocab.index(phrase))
                    i += 2
                    continue

                title_type.append(0)
                title_ind.append(sum_title_vocab.index(title_words[i]))

            i += 1

        assert len(title_ind) == len(title_type)
        return abstract_ind, title_ind,  title_type, len(abstract_ind), len(title_ind)

    def create_batch(self):
        logger.info("Generate Batch size data")
        batch_abstract = []
        batch_title_ind = []
        batch_title_type = []
        abstract_lengths = []
        title_lenghts = []
        for data_list in self.cur_iter:
            # process batch data
            for sample in data_list:
                res = self.generate_ind(sample)
                batch_abstract.append(res[0])
                batch_title_ind.append(res[1])
                batch_title_type.append(res[2])
                abstract_lengths.append(res[3])
                title_lenghts.append(res[4])

                if len(batch_abstract) == self.batch_size:
                    yield batch_abstract, batch_title_ind, batch_title_type, abstract_lengths, title_lenghts
                    batch_abstract = []
                    batch_title_ind = []
                    batch_title_type = []
                    abstract_lengths = []
                    title_lenghts = []

    def create_test(self):
        """
        Generate test data just need abstract and abstract index
        :return:
        """
        logger.info("Generate Testing Data")
        batch_abstract = []
        batch_abstract_txt = []
        batch_title_txt = []
        abstract_lengths = []
        batch_topic = []
        for data_list in self.test_iter:
            for sample in data_list:
                abstract_words = nltk.word_tokenize(sample['abstract'])
                abstract_ind = []
                for word in abstract_words:
                    abstract_ind.append(self.abstract_vocab.index(word))
                batch_abstract.append(abstract_ind)
                abstract_lengths.append(len(abstract_ind))
                batch_abstract_txt.append(sample['abstract'])
                batch_title_txt.append(sample['patent_name'])
                batch_topic.append(sample['topic'])

                if len(batch_abstract) == self.batch_size:
                    yield batch_abstract, batch_abstract_txt, abstract_lengths, batch_title_txt, batch_topic
                    batch_abstract = []
                    batch_abstract_txt = []
                    batch_title_txt = []
                    abstract_lengths = []
                    batch_topic = []


if __name__ == '__main__':
    data_path = "data/demo_data/"
    data_demo = Dataset(data_path, 1)
    print(data_demo.ordinary_vocab)
    print(len(data_demo.abstract_vocab))
    for data in data_demo.create_batch():
        print(data)
