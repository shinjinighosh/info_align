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
    model_path = f"tasks/lex_trans/align_model.chk"
    vis_path = f"tasks/lex_trans/vis"
    params = {"lr": 0.00003, "n_batch": 32}

    model = CountModel(vocab)

    if TRAIN:
        train_count(model, vocab, data, model_path)
    else:
        with open(model_path, "rb") as reader:
            model = pickle.load(reader)

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)


if __name__ == "__main__":
    main()
