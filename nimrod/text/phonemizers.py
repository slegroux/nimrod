# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/text.phonemizer.ipynb.

# %% auto 0
__all__ = ['Phonemizer']

# %% ../../nbs/text.phonemizer.ipynb 4
import platform
import os
if platform.system() == 'Darwin':
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib"

from dotenv import load_dotenv
load_dotenv()

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from phonemizer import phonemize

from typing import List, Tuple, Iterable
from plum import dispatch

# %% ../../nbs/text.phonemizer.ipynb 5
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
    
    @dispatch
    def __call__(self, text:str, n_jobs=1)->str:
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

    @dispatch
    def __call__(self, texts:List[str], n_jobs=1)->List[str]:
        return(
            [phonemize(
                text,
                language=self.language,
                backend=self.backend,
                separator=self.separator,
                strip=self.strip,
                preserve_punctuation=self.preserve_punctuation,
                njobs=n_jobs
                )
        for text in texts])
