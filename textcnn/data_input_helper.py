import collections

import numpy as np
import re

np.random.seed(1235)


def load_word_vector(filepath):
    vocab = collections.OrderedDict()
    word2vec = []
    with open(filepath, 'r', encoding='utf-8') as fr:
        header = next(fr)
        dims = int(header.split(' ')[1])
        vocab['<PAD>'] = 0
        word2vec.append(np.zeros(dims))
        vocab['<UNK>'] = 1
        word2vec.append(np.random.standard_normal(dims))
        idx = 2
        for line in fr:
            line_arr = line.strip().split(' ')
            text = line_arr[0]
            if len(text) == 1:
                vector = np.array([float(x) for x in line_arr[1:]])
                vocab[text] = idx
                word2vec.append(vector)
                idx += 1
    return vocab, np.array(word2vec), dims


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def removezero(x, y):
    nozero = np.nonzero(y)
    print('removezero', np.shape(nozero)[-1], len(y))
    if (np.shape(nozero)[-1] == len(y)):
        return np.array(x), np.array(y)
    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x, y


def read_file_lines(filename, from_size, line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if (i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text


def load_data_and_labels(filepath, max_size=-1):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        data = []
        labels = []
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts[1].strip()) == 0:
                continue
            data.append(parts[1])
            labels.append(parts[0])
        print('data size = ', len(data))
        return [data, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # print('epoch = %d,batch_num = %d,start = %d,end_idx = %d' % (epoch,batch_num,start_index,end_index))
            yield shuffled_data[start_index:end_index]
