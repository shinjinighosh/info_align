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
#from matplotlib import pyplot as plt
import numpy as np
#import seaborn as sns
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

TASK = "lex_trans"
TRAIN = True
COUNT = True
VISUALIZE = False


def main():
    random = np.random.RandomState(0)

    # data, vocab = lex_trans.load_all()
    # test_data, test_vocab = lex_trans.load_test()
    # print(len(data), len(test_data))
    data, vocab = lex_trans.load_toy()
    test_data, test_vocab = lex_trans.load_toy()

    model_path = f"tasks/lex_trans/align_model.chk"
    vis_path = f"tasks/lex_trans/vis"
    params = {"lr": 0.00003, "n_batch": 32}

    train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = CountModel(vocab)

    if TRAIN:
        train_count(model, vocab, data, model_path)
    else:
        with open(model_path, "rb") as reader:
            model = pickle.load(reader)

    print("Finished training/fetching")

    model.eval()
    counts = Counter()

    for i, (src, tgt) in enumerate(data):
        if src == tgt:
            continue
        src_toks = tuple(vocab.decode(src).split())
        tgt_toks = tuple(vocab.decode(tgt).split())
        for (s0, s1), (t0, t1), score in info.parse_greedy(src, tgt, model, vocab):
            counts[src_toks[0][s0:s1], tgt_toks[0][t0:t1]] += score

    print(counts.most_common(50))
    print("Got counts")
    output_file = open("outputs.txt", "w")

    if TASK == "lex_trans":

        translation_dict = {}
        for en, es in test_data:
            en = test_vocab.decode(en)
            es = test_vocab.decode(es)
            if en in translation_dict:
                translation_dict[en].append(es)
            else:
                translation_dict[en] = [es]

        overall_score = 0  # number of words we translated correctly
        seen_words = set()

        for en, es in test_data:
            if test_vocab.decode(en) in seen_words:
                continue
            seen_words.add(test_vocab.decode(en))
            max_score = float("-inf")
            best_translated_split = ("", "",)

            neural_pred = seq_model.sample(formatted_en)  # TODO

            # score splits
            for i in range(1, len(en)):
                max_score_a = float("-inf")
                max_score_b = float("-inf")
                best_translated_split_a = ""
                best_translated_split_b = ""

                split_a = test_vocab.decode(en[:i])
                split_b = test_vocab.decode(en[i:])

                best_split_as = []
                best_split_bs = []

                # choose best translation given split
                for ((key, val), ct) in counts.items():
                    # all ways of translating split_a
                    if key == split_a:
                        score_a = ct

                        if score_a > max_score_a:
                            max_score_a = score_a
                            best_split_as = [val]
                            best_translated_split_a = val

                        elif score_a == max_score_a:  # tie
                            best_split_as.append(val)
                            best_translated_split_a = val

                    # all ways for translating split_b
                    elif key == split_b:
                        score_b = ct

                        if score_b > max_score_b:
                            max_score_b = score_b
                            best_split_bs = [val]
                            best_translated_split_b = val

                        elif score_b == max_score_b:  # tie
                            best_split_bs.append(val)
                            best_translated_split_b = val

                # add scores and get max
                # (independently got best translations for a and b)
                score = max_score_a + max_score_b
                # choose best split
                if score > max_score:
                    max_score = score
                    best_translated_split = (best_translated_split_a, best_translated_split_b)

            translated_a, translated_b = best_translated_split
            translated_word = translated_a + translated_b
            print(test_vocab.decode(es), translated_word, test_vocab.decode(en))

            if translated_word in translation_dict[test_vocab.decode(en)]:
                output_file.write(",".join([test_vocab.decode(
                    es), translated_word, test_vocab.decode(en), str(1)]) + "\n")
                overall_score += 1
                print(overall_score)
            else:
                output_file.write(",".join([test_vocab.decode(
                    es), translated_word, test_vocab.decode(en), str(0)]) + "\n")

    print("Accuracy", overall_score * 100.0 / len(translation_dict))
    output_file.close()

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)


if __name__ == "__main__":
    main()
