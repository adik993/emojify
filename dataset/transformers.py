from typing import List

import torch

import numpy as np
from torchvision.transforms import Compose

from constants import PAD
from glove import Glove


class GloveIndex:
    def __init__(self, glove: Glove):
        self.glove = glove

    def __call__(self, sentence_list: List[str]) -> np.ndarray:
        return np.array([self.glove.to_index(word) for word in sentence_list])


class PadToSizeWithLengthAppended:
    def __init__(self, size):
        self.size = size

    def __call__(self, index_array: np.ndarray) -> np.ndarray:
        array = np.pad(index_array, [0, (self.size + 1) - len(index_array)], mode='constant', constant_values=PAD)
        array[-1] = len(index_array)  # append length
        return array


class ToTensor:
    def __call__(self, array: np.ndarray):
        return torch.tensor(array)


class SentenceToPaddedIndexesWithLength:
    def __init__(self, glove: Glove, max_len):
        self._transformer = Compose((GloveIndex(glove), PadToSizeWithLengthAppended(max_len), ToTensor()))

    def __call__(self, sentence_list: List[str]) -> np.ndarray:
        return self._transformer(sentence_list)
