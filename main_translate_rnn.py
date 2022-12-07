#!/usr/bin/env python3

import numpy as np
import torch
import pickle

from model import Model, PretrainedModel, CountModel, RNNModel
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
        data_padded.append((padded_en, padded_es))

    for en, es in test_data:
        padded_en = en + [-1] * (18 - len(en))
        padded_es = es + [-1] * (18 - len(es))
        test_data_padded.append((padded_en, padded_es))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    train_dataloader = DataLoader(torch.tensor(data_padded), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(torch.tensor(test_data_padded), batch_size=64, shuffle=True)

    model = RNNModel(input_size=18, output_size=18, hidden_dim=12, n_layers=3)
    model = model.to(device)

    n_epochs = 100
    lr = params["lr"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input_seq = input_seq.to(device)
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        #input_seq = input_seq.to(device)
        output, hidden = model(input_seq)
        output = output.to(device)
        target_seq = target_seq.to(device)
        loss = criterion(output, target_seq.view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

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
