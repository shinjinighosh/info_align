import os
import torch

DATA_PATH = f"{os.path.dirname(__file__)}/lex/es/train.txt"


class LexTransVocab:
    def __init__(self, data_path=DATA_PATH):
        vocab = {}
        for special_token in ["<pad>", "<skip1>", "<hole1>", "</s>"]:
            vocab[special_token] = len(vocab)

        self.data_path = data_path
        with open(self.data_path) as reader:
            for line in reader:
                tr, en = line.split()
                for c in list(tr) + list(en):
                    if c not in vocab:
                        vocab[c] = len(vocab)

        self.SKIP1 = vocab["<skip1>"]
        self.HOLE1 = vocab["<hole1>"]
        self.START = self.HOLE1  # vocab["<start>"]
        self.SEP = vocab["</s>"]
        self.END = vocab["</s>"]
        self.PAD = vocab["<pad>"]

        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}

    def encode(self, seq):
        return [self.vocab[c] for c in seq]

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy().tolist()
        seq = [s for s in seq if s != self.PAD]
        return "".join(self.rev_vocab[c] for c in seq)

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)


def load():
    vocab = LexTransVocab()
    data = []
    with open(DATA_PATH) as reader:
        for line in reader:
            tr, en = line.split()
            tr = vocab.encode(tr)
            en = vocab.encode(en)
            data.append((tr, en))
    return data, vocab


def load_test():

    data_path_test = f"{os.path.dirname(__file__)}/lex/es/test.txt"
    vocab = LexTransVocab(data_path_test)
    data = []

    with open(data_path_test) as reader:
        for line in reader:
            tr, en = line.split()
            tr = vocab.encode(tr)
            en = vocab.encode(en)
            data.append((tr, en))
    return data, vocab


def load_all():

    data_path_test = f"{os.path.dirname(__file__)}/lex/es/all.txt"
    vocab = LexTransVocab(data_path_test)
    data = []

    with open(data_path_test) as reader:
        for line in reader:
            tr, en = line.split()
            tr = vocab.encode(tr)
            en = vocab.encode(en)
            data.append((tr, en))
    return data, vocab


def load_toy():

    data_path_test = f"{os.path.dirname(__file__)}/lex/es/toy.txt"
    vocab = LexTransVocab(data_path_test)
    data = []

    with open(data_path_test) as reader:
        for line in reader:
            tr, en = line.split()
            tr = vocab.encode(tr)
            en = vocab.encode(en)
            data.append((tr, en))
    return data, vocab
