# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/text.tokenizers.ipynb.

# %% auto 0
__all__ = ['Phonemizer', 'Tokenizer', 'Numericalizer']

# %% ../../nbs/text.tokenizers.ipynb 4
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from phonemizer import phonemize

# %% ../../nbs/text.tokenizers.ipynb 6
class Phonemizer():
    def __init__(self,
        separator=Separator(word=" ", syllable="|", phone=None), # separator
        language='en-us', # language
        backend='espeak', # phonemization backend (espeak)
        strip=True, # strip
        preserve_punctuation=True # preserve punctuation
        ):
        self.separator = separator
        self.language = language
        self.backend = backend
        self.strip = strip
        self.preserve_punctuation = preserve_punctuation
    
    def __call__(self, text, n_jobs=1):
        return(
            phonemize(
                text,
                language=self.language,
                backend=self.backend,
                separator=self.separator,
                strip=self.strip,
                preserve_punctuation=self.preserve_punctuation,
                njobs=n_jobs
                )
        )

# %% ../../nbs/text.tokenizers.ipynb 12
import torchtext
import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import spacy
from torchtext.datasets import AG_NEWS
from typing import Iterable
from torch.nn.utils.rnn import pad_sequence
# import io

# %% ../../nbs/text.tokenizers.ipynb 13
class Tokenizer:
    def __init__(self, backend='spacy', language='en'):
        self.tokenizer = get_tokenizer(backend, language=language)
        self.counter = Counter()
        self._vocab = None

    def __call__(self, text:str):
        return self.tokenizer(text)
    
    def tokenize_iter(self, data_iter:Iterable):
        for _, text in data_iter:
            yield self.tokenizer(text)


# %% ../../nbs/text.tokenizers.ipynb 15
class Numericalizer():
    def __init__(self, tokenizer:Tokenizer):
        self.tokenizer = tokenizer
        self._vocab = None
    
    def build_map_from_iter(self,data_iter:Iterable, specials = ["<unk>"]):
        self._vocab = build_vocab_from_iterator(self.tokenizer.tokenize_iter(data_iter), specials=specials)
        if "<unk>" in specials:
            self._vocab.set_default_index(self._vocab["<unk>"])
        return self._vocab