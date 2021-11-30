# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random

random.seed(324)


def split(data, split_frac=0.8):
    labels = data['label'].unique()
    train, test = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    for label in labels:
        one_data = data[data['label'] == label]
        data_len = len(one_data)
        mask = np.random.rand(data_len) < split_frac
        one_data_train = one_data[mask]
        one_data_test = one_data[~mask]
        train = pd.concat([train, one_data_train], axis=0)
        test = pd.concat([test, one_data_test], axis=0)
    train.to_csv('../data/train.csv', sep='\t', index=False, columns=None)
    test.to_csv('../data/test.csv', sep='\t', index=False, columns=None)


if __name__ == '__main__':
    columns = ['label', 'text']
    df = pd.read_csv('../data/cutclean_label_corpus10000.txt', sep='\t', names=columns)
    split(df)
