import json
import toolz
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
import cv2
import numpy as np
from typing import Dict, List, Protocol
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.common.registry import Registry
from src.tokenizers.base import BaseTokenizer


def tohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


@Registry.register_tokenizer
class VisualCodebookProcessor(BaseTokenizer):
    name: str = "visual_codebook_tokenizer"
    input_type: str = "image"

    def __init__(self, k_size: int = 32, sample: int = 255, channel=0, num_words: int = 64, **kwargs) -> None:
        super(VisualCodebookProcessor).__init__()
        self.k_size = k_size
        self.sample = sample
        self.channel = channel
        self.num_words = num_words

    def fit(self, images: List[np.ndarray]) -> None:
        """
        Tokenizer that generates a visual bag of words codebook by computing local histograms on the whole dataset.
        Processor clusters the local histograms in order to create regions of feasible words.

        Args:
            images: The list of numpy arrays representing the images.
            k_size: Kernel size of the sliding window that will produce the histogram for comparing with the codebook.
            sample: Local histogram number of bins
            channel: Channel on which we compute the histograms
            num_words: Number of clusters for visual bag of words

        Returns:
            VisualCodeBookProcessor filled with the model to process bag of words
        """
        # TODO: Comments
        self.codebook: List = []
        # TODO: Compute visual codebook efficiently

        for img in tqdm(images):
            img = tohsv(img)
            for i_step in range(0, img.shape[0], self.k_size):
                for j_step in range(0, img.shape[1], self.k_size):
                    hist, _ = np.histogram(
                        img[i_step:i_step+self.k_size, j_step:j_step+self.k_size, self.channel], self.sample)
                    hist = hist / hist.max()
                    self.codebook.append(hist)

        gm = KMeans(self.num_words).fit(self.codebook)
        self.bag_of_visual_words = gm

    def tokenize(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        features = []

        for image in images:
            image_hsv = tohsv(image)
            words_frequency_hist = np.zeros(self.num_words)

            for i_step in range(0, image_hsv.shape[0], self.k_size):
                for j_step in range(0, image_hsv.shape[1], self.k_size):
                    hist, _ = np.histogram(
                        image_hsv[i_step:i_step+self.k_size, j_step:j_step+self.k_size, self.channel], self.sample)
                    value = self.bag_of_visual_words.predict([hist])[0]
                    words_frequency_hist[value] += 1

            features.append(words_frequency_hist /
                            np.max(words_frequency_hist))

        return {
            "result": features
        }


# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """

try:
    from typing import Dict, Iterable, List, Any, Iterator
except ImportError:
    pass


DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


@Registry.register_tokenizer
class BPETokenizer:
    """ 
    Encodes white-space separated text using byte-pair encoding. 

    Taken from https://github.com/soaxelbrooke/python-bpe.
    """
    name: str = "bpe_tokenizer"
    input_type: str = "str"

    def __init__(self, vocab_size=256, pct_bpe=0.2, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=8, required_tokens=None,
                 strict=False, lowercase=True, fixed_length=None,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD, **kwargs):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(
            set(required_tokens or []).union({self.UNK, self.PAD}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max(
            [int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase
        self.fixed_length = fixed_length

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    def byte_pair_counts(self, words):
        # type: (Encoder, Iterable[str]) -> Iterable[Counter]
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()  # type: Counter
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(
                    ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts

    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        word_counts = Counter(word for word in toolz.concat(
            map(self.word_tokenizer, sentences)))
        for token in set(self.required_tokens or []):
            word_counts[token] = int(2**63)
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: Counter
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2**63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(
            vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text):
        # type: (Encoder, Iterable[str]) -> None
        if type(text[0]) is not str:
            _text = [line for s in text for line in s]
        else:
            _text = text

        """ Learn vocab from text. """
        if self.lowercase:
            _text = [l.lower().strip() for l in _text]
        else:
            _text = [l.strip() for l in _text]
        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text)

        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if word not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)

        self.inverse_word_vocab = {
            idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {
            idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        # type: (int, Dict[str, int]) -> None
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize_single_sentence(self, sentence):
        # type: (Encoder, str) -> List[int]
        """ Split a sentence into word and subword tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())

        tokens = []

        for word_token in word_tokens:
            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def tokenize(self, sentences: List[str]):
        # type: (Encoder, str) -> List[List[int]]
        """ Split a sentence into word and subword tokens """
        if type(sentences) is not list and type(sentences) is not tuple:
            sentences = [sentences]

        return self.transform(sentences)

    def transform(self, sentences, reverse=False):
        # type: (Encoder, Iterable[str], bool, int) -> List[List[int]]
        """ Turns space separated tokens into vocab idxs """
        encoded_sentences = []
        direction = -1 if reverse else 1

        for sentence in self._progress_bar(sentences):
            in_subword = False
            encoded = []
            if self.lowercase:
                tokens = list(self.tokenize_single_sentence(sentence.lower().strip()))
            else:
                tokens = list(self.tokenize_single_sentence(sentence.strip()))
            for token in tokens:
                if in_subword:
                    if token in self.bpe_vocab:
                        if token == self.EOW:
                            in_subword = False
                        encoded.append(self.bpe_vocab[token])
                    else:
                        encoded.append(self.word_vocab[self.UNK])
                else:
                    if token == self.SOW:
                        in_subword = True
                        encoded.append(self.bpe_vocab[token])
                    else:
                        if token in self.word_vocab:
                            encoded.append(self.word_vocab[token])
                        else:
                            encoded.append(self.word_vocab[self.UNK])

            if self.fixed_length is not None:
                encoded = encoded[:self.fixed_length]
                while len(encoded) < self.fixed_length:
                    encoded.append(self.word_vocab[self.PAD])

            encoded_sentences.append(encoded[::direction])

        return encoded_sentences

    def inverse_transform(self, rows):
        # type: (Encoder, Iterable[List[int]]) -> Iterator[str]
        """ Turns token indexes back into space-joined text. """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError(
                            'Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError(
                            'Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError(
                            "Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                else:
                    raise ValueError(
                        "Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    def vocabs_to_dict(self, dont_warn=False):
        # type: (Encoder, bool) -> Dict[str, Dict[str, int]]
        """ Turns vocab into dict that is json-serializeable """
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer.  You'll need to reassign it when you load the "
                  "model!")
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'required_tokens': self.required_tokens,
                'strict': self.strict,
                'EOW': self.EOW,
                'SOW': self.SOW,
                'UNK': self.UNK,
                'PAD': self.PAD,
            }
        }

    def save(self, outpath, dont_warn=False, encoding=None, ensure_ascii=True, indent=2):
        # type: (Encoder, str, bool) -> None
        """ Serializes and saves encoder to provided path """
        with open(outpath, 'w', encoding=encoding) as outfile:
            json.dump(self.vocabs_to_dict(dont_warn), outfile,
                      ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, vocabs):
        # type: (Any, Dict[str, Dict[str, int]]) -> Encoder
        """ Load encoder from dict produced with vocabs_to_dict """
        encoder = Encoder(**vocabs['kwargs'])
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']

        encoder.inverse_bpe_vocab = {
            v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {
            v: k for k, v in encoder.word_vocab.items()}

        return encoder

    @classmethod
    def load(cls, in_path):
        # type: (Any, str) -> Encoder
        """ Loads an encoder from path saved with save """
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)
