"""allows to leverage preliminary data prep from lhotse recipes"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/data.lhotse.ipynb.

# %% auto 0
__all__ = ['Encoder', 'Decoder', 'PhonemeCollater']

# %% ../../nbs/data.lhotse.ipynb 5
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

from lhotse import CutSet, RecordingSet, SupervisionSet, Fbank, FbankConfig, MonoCut, NumpyFilesWriter, NumpyHdf5Writer
from lhotse.dataset import BucketingSampler, OnTheFlyFeatures, DynamicBucketingSampler
from lhotse.dataset.collation import TokenCollater
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.vis import plot_batch
from lhotse.recipes import download_librispeech, prepare_librispeech, download_ljspeech, prepare_ljspeech

from typing import Tuple, Dict
import json
import numpy as np

from ..audio.embedding import EncoDecExtractor
from ..text.normalizers import TTSTextNormalizer
from ..text.phonemizers import Phonemizer


# %% ../../nbs/data.lhotse.ipynb 6
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

# %% ../../nbs/data.lhotse.ipynb 28
class PhonemeCollater(TokenCollater):
    def __init__(
            self,  cuts: CutSet,
            add_eos: bool = True,
            add_bos: bool = True,
            pad_symbol: str = "<pad>",
            bos_symbol: str = "<bos>",
            eos_symbol: str = "<eos>",
            unk_symbol: str = "<unk>",
        ):
        super().__init__(
            cuts,
            add_eos=add_eos,
            add_bos=add_bos,
            pad_symbol=pad_symbol,
            bos_symbol=bos_symbol,
            eos_symbol=eos_symbol,
            unk_symbol=unk_symbol
            )
        tokens = {char for cut in cuts for char in cut.custom['phonemes']}
        tokens_unique = (
            [pad_symbol, unk_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(tokens_unique)}
        self.idx2token = [token for token in tokens_unique]
    
    def __call__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor]:
        token_sequences = [" ".join(cut.custom['phonemes']) for cut in cuts]
        max_len = len(max(token_sequences, key=len))
        seqs = [
            ([self.bos_symbol] if self.add_bos else [])
            + list(seq)
            + ([self.eos_symbol] if self.add_eos else [])
            + [self.pad_symbol] * (max_len - len(seq))
            for seq in token_sequences
        ]

        tokens_batch = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq) + int(self.add_eos) + int(self.add_bos)
                for seq in token_sequences
            ]
        )

        return tokens_batch, tokens_lens
