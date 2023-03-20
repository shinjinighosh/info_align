from collections import Counter, defaultdict
import numpy as np
import random
import json
import torch
from torch import nn
from transformers import MT5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import utils

N_HIDDEN = 512
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Estimates (conditional and unconditional) substring probabilities via counting.
# The `observe` functions increment the frequency of the corresponding event.


class CountModel:
    def __init__(self, vocab):
        self.counts = defaultdict(Counter)
        self.totals = Counter()

        self.src_counts = defaultdict(Counter)
        self.src_totals = Counter()
        self.tgt_counts = defaultdict(Counter)
        self.tgt_totals = Counter()
        self.vocab = vocab

    def observe(self, x, y):
        x = tuple(x)
        y = tuple(y)
        self.counts[x][y] += 1
        self.totals[x] += 1

    def observe_src(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.src_counts[x][y] += 1. / scale
        self.src_totals[x] += 1. / scale

    def observe_tgt(self, x, y, scale):
        x = tuple(x)
        y = tuple(y)
        self.tgt_counts[x][y] += 1. / scale
        self.tgt_totals[x] += 1. / scale

    def h_src(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.src_counts[x][y]) - np.log(self.src_totals[x]))

    def h_tgt(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.tgt_counts[x][y]) - np.log(self.tgt_totals[x]))

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, x, y):
        x = tuple(x)
        y = tuple(y)
        return -(np.log(self.counts[x][y]) - np.log(self.totals[x]))


# class TransformerModel():
#     def __init__(self, vocab):
#         super().__init__()
#         self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
#         tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
#         self.vocab = vocab
#
#     def forward(self):
#         # input_ids = tokenizer.encode(text, return_tensors="pt")
#         outputs = model.generate(input_ids)
#         # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return outputs


class PretrainedModel(nn.Module):
    # Estimates substring probabilities by fine-tuning a pre-trained model.
    def __init__(self, vocab):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        self.vocab = vocab
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, out):
        # TODO double-check that this is necessary
        output = self.model(input_ids=inp, decoder_input_ids=out[:, :-1])
        logits = output.logits
        b, l, v = logits.shape
        return self.loss(logits.view(b * l, v), out[:, :-1].reshape(b * l)).view(b, l).sum(dim=1)

    def decode(self, inp):
        return self.model.generate(input_ids=inp, eos_token_id=self.vocab.END)

# Estimates substring probabilities by training a transformer from scratch.


class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), N_HIDDEN)
        self.pos_embedding = nn.Embedding(N_HIDDEN, N_HIDDEN)
        self.transformer = nn.Transformer(N_HIDDEN, batch_first=True)
        self.pred = nn.Linear(N_HIDDEN, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.PAD, reduction="none")

    def forward(self, inp, out):
        out_from = out[:, :-1]
        out_to = out[:, 1:]
        tgt_shape = out_to.shape

        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)
        out_pos = torch.arange(out_from.shape[1], device=out_from.device)[None, :]
        emb_out = self.embedding(out_from) + self.pos_embedding(out_pos)
        mask = nn.Transformer.generate_square_subsequent_mask(out_from.shape[1]).cuda()
        enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
        pred = self.pred(enc)

        pred = pred.reshape(-1, len(self.vocab))
        out_to = out_to.reshape(-1)

        loss = self.loss(pred, out_to)  # .view(tgt_shape) # sperry
        # loss = loss.sum(dim=1) sperry
        return loss

    @torch.no_grad()
    def decode(self, inp, greedy=True):
        inp_pos = torch.arange(inp.shape[1], device=inp.device)[None, :]
        emb_inp = self.embedding(inp) + self.pos_embedding(inp_pos)

        out = torch.tensor([[self.vocab.START]] * inp.shape[0]).cuda()

        for i in range(20):
            out_pos = torch.arange(out.shape[1], device=out.device)[None, :]
            emb_out = self.embedding(out) + self.pos_embedding(out_pos)
            mask = nn.Transformer.generate_square_subsequent_mask(out.shape[1]).cuda()
            enc = self.transformer(emb_inp, emb_out, tgt_mask=mask)
            pred = self.pred(enc)
            if greedy:
                choice = pred[:, -1:].argmax(dim=2)
            else:
                choice = torch.multinomial(torch.exp(pred[:, -1]), 1)
            out = torch.cat((out, choice), dim=1)

        results = []
        for row in out:
            row = row.cpu().numpy().tolist()
            if self.vocab.END in row:
                row = row[:row.index(self.vocab.END) + 1]
            results.append(row)
        return results


# class RNNModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(RNNModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_size)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         hidden = self.init_hidden(batch_size)
#         out, hidden = self.rnn(x, hidden)
#         out = out.contiguous().view(-1, self.hidden_dim)
#         out = self.fc(out)
#
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         hidden = torch.zeros(self.n_layers, self.hidden_dim).to(device)
#         return hidden


# class LSTMModel(nn.Module):
#
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(LSTMModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers


# class LSTM1(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
#         super(LSTM1, self).__init__()
#         self.num_classes = num_classes  # number of classes
#         self.num_layers = num_layers  # number of layers
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # hidden state
#         self.seq_length = seq_length  # sequence length
#
#         self.lstm_encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                                     num_layers=num_layers, batch_first=True)  # lstm
#
#         self.lstm_decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                                     num_layers=num_layers, batch_first=True)  # lstm
#
#         self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
#         self.fc = nn.Linear(128, num_classes)  # fully connected last layer
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
#                           requires_grad=True)  # hidden state
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
#                           requires_grad=True)  # internal state
#         # Propagate input through LSTM
#         # lstm with input, hidden, and internal state
#         output_encoder, (hn, cn) = self.lstm_encoder(x, (h_0, c_0))
#
#         output_decoder, (_, _) = self.lstm_decoder(output_encoder, (hn, cn))
#
#         hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
#         out = self.relu(hn)
#         out = self.fc_1(out)  # first Dense
#         out = self.relu(out)  # relu
#         out = self.fc(out)  # Final Output
#         return out

# class Encoder1(nn.Module):
#     def __init__(self, input_dim=44, hid_dim=512):
#         super().__init__()
#
#         self.onehot_func = lambda x: torch.stack([torch.stack(
#             [nn.functional.one_hot(torch.cat((torch.tensor([0]), a + 2)), 44) for a in b]) for b in x])
#
#         embedding_size = 16
#         self.embedding = nn.Embedding(input_dim, embedding_size).to(torch.float32)
#         self.rnn = nn.LSTM(embedding_size, hid_dim, batch_first=True)
#
#     def forward(self, src):
#         embedded = torch.stack([self.embedding(x.to(torch.int64)) for x in src])
#         print(embedded.shape)
#         outputs, (hidden, cell) = self.rnn(embedded)
#         return hidden, cell
#
#
# class Decoder1(nn.Module):
#     def __init__(self, output_dim=44, hid_dim=512):
#         super().__init__()
#
#         embedding_dim = 16
#         self.embedding = nn.Embedding(output_dim, embedding_dim).to(torch.float32)
#         self.rnn = nn.LSTM(output_dim, hid_dim, batch_first=True)
#         self.fc_out = nn.Linear(hid_dim, output_dim)
#
#     def forward(self, input, hidden, cell):
#
#         embedded = self.embedding(input.to(torch.int64)).unsqueeze(1)
#         output, (hidden, cell) = self.rnn(input, (hidden, cell))
#         prediction = self.fc_out(output.squeeze(1))
#
#         return prediction
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = trg.shape[0]
#         max_len = trg.shape[1]
#         # trg_vocab_size = trg.shape[2]
#         # trg_vocab_size =
#
#         # outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(src.device)
#         hidden, cell = self.encoder(src)
#
#         input = trg[:, 0:1, :]
#         # print(input.shape)
#         # print(input)
#         for t in range(1, max_len):
#             output = self.decoder(input, hidden, cell)
#             # print(output)
#             outputs[:, t:t + 1, :] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             top1 = top1.reshape(-1, 1, trg_vocab_size)
#             input = trg[:, t:t + 1, :] if teacher_force else top1.to(torch.float32)
#
#         return teacher_force, outputs

#
# # Define hyperparameters
# INPUT_DIM = 10000
# OUTPUT_DIM = 10000
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 256
#


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        assert n_layers == 1
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)
        self.attention_write = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, tgt, state):
        h, c = state
        h = h.squeeze(0)
        c = c.squeeze(0)
        hiddens = []
        for i in range(tgt.shape[0]):
            h, c = self.cell(tgt[i], (h, c))
            key = self.attention_key(h)
            scores = (src * key.unsqueeze(0)).sum(dim=1, keepdim=True)
            weights = scores.softmax(dim=0)
            pooled = (src * weights).sum(dim=0)
            h = h + self.attention_write(pooled)
            hiddens.append(h)

        hiddens = torch.stack(hiddens)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        return hiddens, (h, c)


# Ordinary neural sequence model for comparison
class SequenceModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 32)
        self.enc = nn.LSTM(32, 256, 1)
        #self.dec = nn.LSTM(32, 256, 1)
        self.dec = LSTMWithAttention(32, 256, 1)
        self.pred = nn.Linear(256, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.PAD)
        self.vocab = vocab

    def compute_log_likelihood(self, inp_word, tgt_word, max_len=20):
        inp_emb = self.emb(inp_word.unsqueeze(1))  # might not need unsqueeze
        inp_enc, state = self.enc(inp_emb)
        # out = torch.ones(1, 1).long() * self.vocab.START
        nll = 0

        for i in range(max_len):
            # out_emb = self.emb(out[-1:, :])
            out_emb = self.emb(tgt_word[i].unsqueeze(0).unsqueeze(0))
            hiddens, state = self.dec(inp_enc, out_emb, state)
            pred = self.pred(hiddens).squeeze(0)
            pred = (pred / .1).softmax(dim=1)
            # need to make sure that first letter in tgt_word is START token
            nll += -np.log(pred[0, tgt_word[i]].detach().cpu().numpy().item())

        return nll

    def sample(self, inp, max_len=20):
        inp_emb = self.emb(inp)
        inp_enc, state = self.enc(inp_emb)
        n_batch = inp.shape[1]
        out = torch.ones(1, n_batch).long() * self.vocab.START
        for i in range(max_len):
            out_emb = self.emb(out[-1:, :])
            hiddens, state = self.dec(inp_enc, out_emb, state)
            pred = self.pred(hiddens).squeeze(0)
            pred = (pred / .1).softmax(dim=1)
            # samp = torch.multinomial(pred, num_samples=1)
            samp = torch.argmax(pred).unsqueeze(0).unsqueeze(0)
            out = torch.cat([out, samp], dim=0)
            # import pdb
            # pdb.set_trace()

        results = []
        for i in range(n_batch):
            seq = out[:, i].detach().cpu().numpy().tolist()
            if self.vocab.END in seq:
                seq = seq[:seq.index(self.vocab.END) + 1]
            results.append(seq)
        return results

    def forward(self, inp, out):
        out_src = out[:-1, :]
        out_tgt = out[1:, :]

        inp_emb = self.emb(inp)
        out_emb = self.emb(out_src)

        inp_enc, state = self.enc(inp_emb)
        hiddens, _ = self.dec(inp_enc, out_emb, state)
        pred = self.pred(hiddens)

        pred = pred.view(-1, len(self.vocab))
        out_tgt = out_tgt.view(-1)

        loss = self.loss(pred, out_tgt)
        return loss, (pred, out_tgt)


class CountModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CountModel):
            counts = [(k, list(v.items())) for k, v in obj.counts.items()]
            src_counts = [(k, list(v.items())) for k, v in obj.src_counts.items()]
            tgt_counts = [(k, list(v.items())) for k, v in obj.tgt_counts.items()]

            return {
                "_cls": "CountModel",
                "counts": counts,
                "totals": list(obj.totals.items()),
                "src_counts": src_counts,
                "src_totals": list(obj.src_totals.items()),
                "tgt_counts": tgt_counts,
                "tgt_totals": list(obj.tgt_totals.items()),
            }
        else:
            return super().default(obj)


def _tuplize(seq):
    if isinstance(seq, list):
        return tuple(_tuplize(s) for s in seq)
    return seq


def decode_count_model(obj):
    if "_cls" not in obj:
        return obj
    if obj["_cls"] == "CountModel":
        model = CountModel(None)
        model.counts = {k: dict(v) for k, v in _tuplize(obj["counts"])}
        model.totals = dict(_tuplize(obj["totals"]))
        model.src_counts = {k: dict(v) for k, v in _tuplize(obj["src_counts"])}
        model.src_totals = dict(_tuplize(obj["src_totals"]))
        model.tgt_counts = {k: dict(v) for k, v in _tuplize(obj["tgt_counts"])}
        model.tgt_totals = dict(_tuplize(obj["tgt_totals"]))
        return model
    assert False
