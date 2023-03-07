#!/usr/bin/env python3

import numpy as np
import torch
import pickle

from model import Model, PretrainedModel, CountModel, RNNModel, Encoder1, Decoder1, Seq2Seq
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
from torch import nn
import torch.optim as optim
from tqdm import tqdm


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
    # print(vocab)
    test_data, test_vocab = lex_trans.load_test()
    print(len(data), len(test_data))

    # max_en_letter = 0
    # for en, es in data:
    #     max_en_letter = max((max_en_letter, max(en)))
    # print(max_en_letter)

    # data, vocab = lex_trans.load_toy()
    # test_data, test_vocab = lex_trans.load_toy()
    model_path = f"tasks/lex_trans/align_model_rnn.chk"
    vis_path = f"tasks/lex_trans/vis"
    params = {"lr": 0.000003, "n_batch": 32}

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

    import pdb

    # Instantiate models
    encoder = Encoder1()
    decoder = Decoder1()
    model = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-2)

    def one_hotty(x): return torch.stack([torch.stack(
        [nn.functional.one_hot(torch.cat((torch.tensor([0]), a + 2)), 44) for a in b]) for b in x])

    def train(model, batch_loader, optimize=None, criterion=nn.CrossEntropyLoss(), clip=None):

        epoch_loss = 0

        for i, batch in tqdm(enumerate(batch_loader)):

            optimizer.zero_grad()

            # print(batch.permute(1, 0, 2).shape)
            src, trg = batch.permute(1, 0, 2)
            # print(src.shape, trg.shape)
            # trg_max = trg.argmax(1)

            # src, trg = one_hotty(batch).permute(1, 0, 2, 3).to(torch.float32)
            # trg_max = trg.argmax(2)

            teacher_force, output = model(src, trg)

            output_dim = output.shape[-1]
            # output = output[:, 1:, :].view(-1, output_dim)
            # trg = trg[:, 1:, :].view(-1)

            # print(output[:, 0, :].shape)
            # print(trg.argmax(2).shape)

            loss = 0

            # trg_max[:, 1] = 4
            # trg_max[:, 2] = 5

            for i in range(3):

                loss += criterion(output[:, i, :], trg_max[:, i])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            epoch_loss += loss.item()

        print(output.argmax(2)[0])
        print(trg_max[0])

        return epoch_loss / len(batch_loader)

    for i in range(1000):

        # print(torch.stack([torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])] * 64
        #                   ))
        results = train(model, train_dataloader)
        print(i, results)

    pdb.set_trace()

    # model = RNNModel(input_size=1, output_size=1, hidden_dim=12, n_layers=3)
    # model = model.to(device)
    #
    # n_epochs = 300
    # lr = params["lr"]
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # # input_seq = input_seq.to(device)
    # for epoch in range(1, n_epochs + 1):
    #     for batch_idx, misshaped_data in enumerate(train_dataloader):
    #         data_1 = []
    #         target_1 = []
    #         for example_inp, example_op in misshaped_data:
    #             data_1.append(torch.stack([torch.tensor([x])
    #                                        for x in example_inp]).type(torch.FloatTensor))
    #             target_1.append(torch.stack([torch.tensor([x])
    #                                          for x in example_op]).type(torch.FloatTensor))
    #         data_1 = torch.stack(data_1).to(device)
    #         target_1 = torch.stack(target_1).to(device)
    #         print(data_1.shape, target_1.shape)
    #         optimizer.zero_grad()  # Clears existing gradients from previous epoch
    #         output, hidden = model(data_1)
    #         output = output.to(device)
    #         # print(output.shape, hidden.shape)
    #         target_1 = target_1.to(device)
    #         loss = criterion(output, target_1)
    #         loss.backward()
    #         optimizer.step()
    #
    #     if epoch % 10 == 0:
    #         print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    #         print("Loss: {:.4f}".format(loss.item()))

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
