import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from constants import PAD
from glove import Glove


class GloveEmbedding(nn.Module):
    def __init__(self, glove: Glove, requires_grad=False):
        super().__init__()
        self.embedding = nn.Embedding(glove.vocabulary_size() + 1, glove.embedding_dim(), padding_idx=PAD)
        self.embedding.weight = nn.Parameter(torch.tensor(glove.to_matrix()))
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, x):
        return self.embedding(x)


class Emojify(nn.Module):
    def __init__(self, glove: Glove, lstm_hidden_size=128, lstm_layers=2, n_classes=5):
        super().__init__()
        self.embedding = GloveEmbedding(glove)
        self.lstm = nn.LSTM(input_size=glove.embedding_dim(),
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            num_layers=lstm_layers,
                            dropout=0.5)
        self.linear = nn.Linear(lstm_hidden_size, n_classes)

    def forward(self, x_with_lengths: torch.Tensor):
        x = x_with_lengths[:, :-1]
        total_length = x.size(-1)
        lengths = x_with_lengths[:, -1]
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=PAD, total_length=total_length)
        x = x[torch.arange(x.size(0)), lengths - 1, :]  # it's many to one RNN, so we need to take last element of the sequence
        x = self.linear(x)
        return x

    def summary(self):
        print(self)
        parameters_count = sum(p.numel() for p in self.parameters())
        trainable_parameters_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Trainable parameters: {:,}'.format(trainable_parameters_count))
        print('Total parameters: {:,}'.format(parameters_count))
