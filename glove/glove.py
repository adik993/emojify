from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlretrieve
from progressbar import ProgressBar, Percentage, Bar

import torch

import numpy as np

from constants import PAD


class Glove:
    def __init__(self, file_path='data/glove.6B.50d.txt', url='https://nlp.stanford.edu/data/glove.6B.zip'):
        self._download_if_necessary(file_path, url)
        self._word_to_vec = self._load(file_path)
        self._word_to_index = {word: i + 1 for i, word in enumerate(sorted(self._word_to_vec.keys()))}
        self._index_to_word = {i + 1: word for i, word in enumerate(sorted(self._word_to_vec.keys()))}
        assert PAD == 0
        self._e = np.array(
            [np.zeros(self.embedding_dim())] + [self._word_to_vec[key] for key in sorted(self._word_to_vec.keys())],
            dtype=np.float32)

    @staticmethod
    def _download_if_necessary(path, url):
        file_path = Path(path)
        if not file_path.exists():
            print(f'file {file_path} does not exist. Downloading it from {url}...')
            progress_bar = ProgressBar(widgets=[Percentage(), Bar()], min_value=0, max_value=100, poll_interval=1)
            zip_filename, _ = urlretrieve(url, reporthook=Glove._download_progress_hook(progress_bar))
            print('unzipping...')
            with ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(file_path.parent)
            print(f'download complete')

    @staticmethod
    def _download_progress_hook(progress_bar: ProgressBar):
        def progress(count, block_size, total_size):
            percentage = int(count * block_size * 100 / total_size)
            force = percentage == 100
            progress_bar.update(percentage, force)

        return progress

    @staticmethod
    def _load(file_path):
        with open(file_path) as f:
            word_to_vec = {}
            for line in f:
                word, *vector_str = line.split(r' ')
                word_to_vec[word] = np.asarray(vector_str, np.float32)
            return word_to_vec

    def to_index(self, word: str) -> int:
        return self._word_to_index[word]

    def to_word(self, index: int) -> str:
        return self._index_to_word[index]

    def vocabulary_size(self):
        return len(self._word_to_vec)

    def embedding_dim(self):
        return next(iter(self._word_to_vec.values())).shape[0]

    def to_matrix(self):
        return self._e

    def to_sentences(self, data: torch.Tensor):
        return [self.to_sentence(row) for row in data]

    def to_sentence(self, row: torch.Tensor):
        return ' '.join([self.to_word(element.item()) for element in row if element.item() != 0])

    def summary(self):
        print('Vocabulary size:', self.vocabulary_size())
        print('Embedding vector dimension:', next(iter(self._word_to_vec.values())).shape)
        print('Embedding matrix shape:', self.to_matrix().shape)
