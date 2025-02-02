#!/usr/bin/env python3

import numpy as np
import torch
import pickle
import bz2
import json

from model import Model, PretrainedModel, CountModel, SequenceModel, decode_count_model
from trainer import train, train_count, train_seq
import info
import utils
from visualize import visualize
from torch.nn.utils.rnn import pad_sequence


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
TO_EVAL = True

# TASK = "cogs"
# TRAIN = False
# COUNT = True
# VISUALIZE = True


def main():
    random = np.random.RandomState(0)

    if TASK == "lex_trans":
        from tasks import lex_trans

        # data, vocab = lex_trans.load()

        data, vocab = lex_trans.load_all()
        test_data, test_vocab = lex_trans.load_test()
        # test_data, test_vocab = lex_trans.load_all()

        model_path = f"tasks/lex_trans/align_model.chk"
        seq_path = f"tasks/lex_trans/seq_model.chk"
        vis_path = f"tasks/lex_trans/vis"
        params = {"lr": 0.00003, "n_batch": 32}
        seq_params = {"lr": 0.003, "n_batch": 32}

    elif TASK == "cogs":
        from tasks import cogs
        data, vocab = cogs.load()
        model_path = f"tasks/cogs/align_model.chk"
        seq_path = f"tasks/cogs/seq_model.chk"
        vis_path = "tasks/cogs/vis"
        params = {"lr": 0.00003, "n_batch": 32}

    if COUNT:
        model = CountModel(vocab)
    else:
        model = Model(vocab).cuda()

    seq_model = SequenceModel(vocab)

    if TRAIN:
        # if COUNT:
        #    train_count(model, vocab, data, model_path)
        # else:
        #    train(model, vocab, data, model_path, random, params)

        trained_model = train_seq(seq_model, vocab, data, seq_path,
                                  random, seq_params, eval_data=test_data)
        print("training finished")

    else:
        if COUNT:
            # with open(model_path, "rb") as reader:
            #    model = pickle.load(reader)
            with bz2.open(model_path, "rt", encoding="utf-8") as reader:
                model = json.load(reader, object_hook=decode_count_model)
                model.vocab = vocab
        else:
            model.load_state_dict(torch.load(model_path))

    if VISUALIZE:
        visualize(model, vocab, data, vis_path)

    # evaluation
    output_file = open("outputs_neural.txt", "w")

    if TASK == "lex_trans" and TO_EVAL:

        translation_dict = {}
        for en, es in test_data:
            en = vocab.decode(en)
            es = vocab.decode(es)
            if en in translation_dict:
                translation_dict[en].append(es)
            else:
                translation_dict[en] = [es]

        overall_score = 0  # number of words we translated correctly
        seen_words = set()

        for en, es in test_data:
            if vocab.decode(en) in seen_words:
                continue
            seen_words.add(vocab.decode(en))

            inp = [torch.tensor([vocab.START] + en + [vocab.END])]
            padded_inp = pad_sequence(inp, padding_value=vocab.PAD)
            translated_word, = seq_model.sample(padded_inp)

            # import pdb
            # pdb.set_trace()

            # print(test_vocab.decode(en), test_vocab.decode(translated_word)[7:-4])

            if vocab.decode(translated_word)[7:-4] in translation_dict[vocab.decode(en)]:

                output_file.write(",".join([vocab.decode(
                    es), vocab.decode(translated_word)[7:-4], vocab.decode(en), str(1)]) + "\n")
                print(",".join([vocab.decode(
                    es), vocab.decode(translated_word)[7:-4], vocab.decode(en), str(1)]))
                overall_score += 1
                print(overall_score)
            else:
                output_file.write(",".join([vocab.decode(
                    es), vocab.decode(translated_word)[7:-4], vocab.decode(en), str(0)]) + "\n")

        print("Accuracy", overall_score * 100.0 / len(translation_dict))
        output_file.close()


if __name__ == "__main__":
    main()
