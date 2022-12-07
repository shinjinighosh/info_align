#!/usr/bin/env python3

import numpy as np
import torch
import pickle

from model import Model, PretrainedModel, CountModel
from trainer import train, train_count
import info
import utils
from visualize import visualize
from tasks import lex_trans
from collections import Counter
# from matplotlib import pyplot as plt
import numpy as np
# import seaborn as sns
import torch
from torch.utils.data import DataLoader


# configuration options
# task: what task to perform (either lex_trans [word-level translation] or
#        cogs [semantic parsing])
# train: if true, trains a new model from scratch; if false, loads one from disk
# count: what kind of model to train: if true, trains a count-based model
#        (only do this for lex_trans!); if false, trains a neural masked LM
# visualize: if true, runs a visualization step that writes model predictions
#        to an html file


def main():
    random = np.random.RandomState(0)

    data, vocab = lex_trans.load_all()
    test_data, test_vocab = lex_trans.load_test()
    print(len(data), len(test_data))
    # data, vocab = lex_trans.load_toy()
    # test_data, test_vocab = lex_trans.load_toy()
    model_path = f"tasks/lex_trans/align_model_rnn.chk"
    vis_path = f"tasks/lex_trans/vis"
    params = {"lr": 0.00003, "n_batch": 32}

    data_padded = []
    test_data_padded = []

    for en, es in data:
        padded_en = en + [-1] * (18 - len(en))
        padded_es = es + [-1] * (18 - len(es))
        data_padded.append((padded_en, padded_es,))

    for en, es in test_data:
        padded_en = en + [-1] * (18 - len(en))
        padded_es = es + [-1] * (18 - len(es))
        test_data_padded.append((padded_en, padded_es,))

    # for (word1, word2) in data:
    #     print(vocab.decode(word1))
    #     print(vocab.decode(word2))

    train_dataloader = DataLoader(np.array(data_padded), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(np.array(test_data_padded), batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # for understanding
    # for i, (src, tgt) in enumerate(data):
    #     if src == tgt:
    #         continue
    #     src_toks = tuple(vocab.decode(src).split())
    #     tgt_toks = tuple(vocab.decode(tgt).split())
    #     for (s0, s1), (t0, t1), score in info.parse_greedy(src, tgt, model, vocab):
    #         counts[src_toks[0][s0:s1], tgt_toks[0][t0:t1]] += score


if __name__ == "__main__":
    main()
