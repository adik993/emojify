import torch

from torch.utils.data import Dataset, TensorDataset
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import default_collate

from dataset.transformers import GloveIndex, PadToSizeWithLengthAppended, SentenceToPaddedIndexesWithLength
from glove import Glove


class EmojifyDataset(Dataset):
    def __init__(self, file_path, glove: Glove):
        self._data = pd.read_csv(file_path, names=['sentence', 'emoji'], dtype={'sentence': str, 'emoji': np.int64})
        self._data['sentence'] = self._data['sentence'].str.strip().str.lower().str.split()
        self._transformer = SentenceToPaddedIndexesWithLength(glove, self._data['sentence'].str.len().max())

    def sentence_to_tensor(self, sentence, device) -> torch.Tensor:
        return self._transformer(sentence.strip().lower().split()).to(device)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data.iloc[index]
        x = self._transformer(row['sentence'])
        y = torch.tensor(row['emoji'], dtype=torch.long)
        return x, y

    @staticmethod
    def collate_fn(batch):
        x_with_lengths, y = default_collate(batch)
        lengths = x_with_lengths[:, -1]
        sorted_indexes = torch.argsort(lengths, dim=0, descending=True)
        return x_with_lengths[sorted_indexes], y[sorted_indexes]
