from collections import Counter, defaultdict
import numpy as np
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

        loss = self.loss(pred, out_to).view(tgt_shape)
        loss = loss.sum(dim=1)
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


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, self.hidden_dim).to(device)
        return hidden


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

class Encoder1(nn.Module):
    def __init__(self, input_dim=44, hid_dim=128):
        super().__init__()

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.onehot_func = lambda x: torch.stack([torch.stack(
            [nn.functional.one_hot(torch.cat((torch.tensor([0]), a + 2)), 44) for a in b]) for b in x])

        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first=True)

    def forward(self, src):
        # embedded = self.embedding(src)
        print("hi")
        outputs, (hidden, cell) = self.rnn(src)
        return hidden, cell
#
#
# class DecoderGPT(nn.Module):
#     def __init__(self, output_dim, emb_dim, hid_dim):
#         super().__init__()
#
#         self.embedding = nn.Embedding(output_dim, emb_dim)
#         self.rnn = nn.LSTM(emb_dim, hid_dim)
#         self.fc_out = nn.Linear(hid_dim, output_dim)
#
#     def forward(self, input, hidden, cell):
#         input = input.unsqueeze(0)
#         embedded = self.embedding(input)
#         output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
#         prediction = self.fc_out(output.squeeze(0))
#
#         return prediction, hidden, cell
#
#
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
# import
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size = trg.shape[1]
#         max_len = trg.shape[0]
#         trg_vocab_size = self.decoder.fc_out.out_features
#
#         outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(src.device)
#         hidden, cell = self.encoder(src)
#
#         input = trg[0, :]
#         for t in range(1, max_len):
#             output, hidden, cell = self.decoder(input, hidden, cell)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             input = trg[t] if teacher_force else top1
#
#         return outputs
#
#
# # Define hyperparameters
# INPUT_DIM = 10000
# OUTPUT_DIM = 10000
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 256
#
