# -*- encoding:utf-8 -*_
"""
Version2:
利用topic rank生成关键词
代码思路：
1. 数据集中每一条sample有：abstract, patent_name, topic_phrase
2. 将数据集按照比例随机分成两份，train和test
"""
import os
import random
import json
import re
from pytopicrank import TopicRank
import nltk
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def generate_topic(data_path):
    """
    Have these following functions:
    1. generate top 10 topic phrase for each sample
    2. remove samples that have can't readable char
    :param data_path: all data
    :return:
    """
    file_list = os.listdir(data_path)
    file_nums = len(file_list)
    random.shuffle(file_list)
    for i in range(file_nums):
        file_name = data_path + "/" + file_list[i]
        new_name = "data/raw/" + "{}.json".format(i)
        f_new = open(new_name,'w')
        data_list = []
        with open(file_name) as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                    abstract = data['abstract'].lower()
                    data_title = data['patent_name'].lower()
                    title_word = nltk.word_tokenize(data_title)
                    if len(title_word) < 5:
                        continue
                    abstract = abstract.replace(" &# 39 ; s ", ' ')
                    abstract = re.sub(r'[^a-z\s]', '', abstract)
                    word_list = nltk.word_tokenize(abstract)
                    word_list = [lemma.lemmatize(item) for item in word_list]
                    title_word = [lemma.lemmatize(item) for item in title_word]
                    abstract = " ".join(word_list)
                    data_title = " ".join(title_word)
                    data['abstract'] = abstract
                    data['patent_name'] = data_title
                    tp = TopicRank(abstract)
                    data['topic'] = tp.get_top_n(n=10)
                    if len(data['topic']) < 10:
                        continue
                    data_list.append(data)
                except Exception as e:
                    continue
        f_new.write(json.dumps(data_list))
        f_new.close()


def split_dataset(raw_path, dst_path):
    """
    The ratio is 1:9
    :param raw_path:
    :return:
    """
    file_list = os.listdir(raw_path)
    file_nums = len(file_list)
    random.shuffle(file_list)
    test = file_nums // 9
    for i in range(file_nums):
        old_name = raw_path + file_list[i]
        if i < test:
            new_name = dst_path + "test.{}.json".format(i+1)
        else:
            new_name = dst_path + "train.{}.json".format(i-test+1)

        os.rename(old_name, new_name)



if __name__ == '__main__':
    raw_path = "data/raw/"
    dst_path = "data/patent/"
    #generate_topic(raw_path)
    split_dataset(raw_path, dst_path)
