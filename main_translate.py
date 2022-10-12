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
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch


# configuration options
# task: what task to perform (either lex_trans [word-level translation] or
#        cogs [semantic parsing])
# train: if true, trains a new model from scratch; if false, loads one from disk
# count: what kind of model to train: if true, trains a count-based model
#        (only do this for lex_trans!); if false, trains a neural masked LM
# visualize: if true, runs a visualization step that writes model predictions
#        to an html file

TASK = "lex_trans"
TRAIN = True
COUNT = True
VISUALIZE = False


def main():
    random = np.random.RandomState(0)

    data, vocab = lex_trans.load()
    test_data, test_vocab = lex_trans.load_test()
    model_path = f"tasks/lex_trans/align_model_shin.chk"
    vis_path = f"tasks/lex_trans/vis"
    params = {"lr": 0.00003, "n_batch": 32}

    model = CountModel(vocab)

    if TRAIN:
        train_count(model, vocab, data, model_path)
    else:
        with open(model_path, "rb") as reader:
            model = pickle.load(reader)

    model.eval()
    counts = Counter()

    for i, (src, tgt) in enumerate(data):
        if src == tgt:
            continue
        src_toks = tuple(vocab.decode(src).split())
        tgt_toks = tuple(vocab.decode(tgt).split())

        for (s0, s1), (t0, t1), score in info.parse_greedy(src, tgt, model, vocab):
            counts[src_toks[s0:s1], tgt_toks[t0:t1]] += score

    if TASK == "lex_trans":
        overall_score = 0

        for tr, en in test_data:
            max_score = 0
            max_score_split = None
            for i in range(1, len(en)):
                split_a = en[:i]
                split_b = en[i:]
                # score splits, recheck
                score = counts[split_a, split_b]
                if score > max_score:
                    max_score = score
                    max_score_split = (split_a, split_b)

            max_split_a = max_score_split[0]
            max_split_b = max_score_split[1]

            translated_a = vocab[max_split_a]
            translated_b = vocab[max_split_b]

            translated_word = translated_a + translated_b
            # score translated word, but how?
            if translated_word == tr:
                overall_score += 1

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)


if __name__ == "__main__":
    main()
