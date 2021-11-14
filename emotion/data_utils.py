from typing import List, Optional, Tuple, Iterator

import numpy as np
import torch

import texar.torch as tx
from texar.torch.hyperparams import HParams
import random
import json
Example = Tuple[np.ndarray, np.ndarray]

from nltk.tokenize import word_tokenize
import math
import numpy as np
from modules import EmotionVocab
def get_lr_multiplier(step: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    square-root decay.
    """
    multiplier = (min(1.0, step / warmup_steps) *
                  (1 / math.sqrt(max(step, warmup_steps))))
    return multiplier

class CustomBatchingStrategy(tx.data.BatchingStrategy[Example]):
    r"""Create dynamically-sized batches for paired text data so that the total
    number of source and target tokens (including padding) inside each batch is
    constrained.
    Args:
        max_tokens (int): The maximum number of source or target tokens inside
            each batch.
    """
    max_src_len: int
    max_tgt_len: int
    cur_batch_size: int

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def reset_batch(self) -> None:
        self.max_src_len = 0
        self.max_tgt_len = 0
        self.cur_batch_size = 0

    def add_example(self, ex: Example) -> bool:
        max_src_len = max(self.max_src_len, len(ex['src_ids']))
        max_tgt_len = max(self.max_tgt_len, len(ex['tgt_ids']))
        if ((self.cur_batch_size + 1) *
                max(max_src_len, max_tgt_len) > self.max_tokens):
            return False
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.cur_batch_size += 1
        return True

class TextLineDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    example = json.loads(line.strip())
                    for i in range(len(example['utters'])//2):
                        j = i*2
                        src_length = sum([len(u) for u in example['utters'][:j+1]])
                        if src_length > 512 or len(example['utters'])<2:
                            print(f"long sentence: {src_length}")
                            continue
                        yield {'utters': example['utters'][:j+2], 'emotion': example['context'], 'emotion_cause': example['emotion_cause']}
                        
class EvalDataSource(tx.data.TextLineDataSource):
    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    example = json.loads(line.strip())
                    if len(example['utters']) % 2 != 0:
                        example['utters'] = example['utters'][:-1]
                    src_length = sum([len(u) for u in example['utters']])
                    if src_length > 512 or len(example['utters']) < 2:
                        print(f"long sentence: {src_length}")
                        continue
                    yield {'utters': example['utters'], 'emotion': example['context'], 'emotion_cause': example['emotion_cause']}
                    


class TrainData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = TextLineDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self._vocab = tx.data.Vocab(self._hparams.dataset.vocab_file)
        self._emotion_vocab = EmotionVocab(self._hparams.dataset.emotion_file)
     
        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None, 
                        'vocab_file':None,
                        'emotion_file':None},
        }

    def process(self, raw_example):
        
        utters = raw_example['utters']
        emotion = raw_example['emotion']
        emotion_cause = raw_example['emotion_cause']
        src = []
        cause_ids = [0.0]
        user_ids = [0]
        for idx,u in enumerate(utters[:-1]):
            if idx in emotion_cause:
                cause_ids += [1] * (len(u)+1)
            else:
                cause_ids += [0] * (len(u)+1)
            if idx % 2 == 0:
                user_ids += [1] * (len(u)+1)
            else:
                user_ids += [2] * (len(u)+1)
            src.append(' '.join(u+['<SEP>']))
        # src = [' '.join(u + ['<SEP>']) for u in utters[:-1]]
        src = ' '.join(['<CLS>'] + src)
        tgt = ' '.join(['<BOS>']+utters[-1]+['<EOS>'])
        return {
            "src_text": src,
            "src_ids": self._vocab.map_tokens_to_ids_py(src.split(' ')),
            "tgt_text": tgt,
            "tgt_ids": self._vocab.map_tokens_to_ids_py(tgt.split(' ')),
            "emotion_text": emotion,
            "emotion_id": [self._emotion_vocab.token_to_id_map_py[emotion]],
            "cause_ids": np.array(cause_ids),
            "user_ids": np.array(user_ids)
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        src_text = [ex["src_text"] for ex in examples]
        src_ids, src_lengths = tx.data.padded_batch(
            [ex["src_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        tgt_text = [ex["tgt_text"] for ex in examples]
        tgt_ids, tgt_lengths = tx.data.padded_batch(
            [ex["tgt_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        cause_ids, cause_lengths = tx.data.padded_batch(
            [ex["cause_ids"] for ex in examples], pad_value=0.0)
        user_ids, user_lengths = tx.data.padded_batch(
            [ex["user_ids"] for ex in examples], pad_value=0)

        emotion_text = [ex["emotion_text"] for ex in examples]
        emotion_id = np.array([ex["emotion_id"] for ex in examples])
        return tx.data.Batch(
            len(examples),
            src_text=src_text,
            src_text_ids=torch.from_numpy(src_ids),
            src_lengths=torch.tensor(src_lengths),
            tgt_text=tgt_text,
            tgt_text_ids=torch.from_numpy(tgt_ids),
            tgt_lengths=torch.tensor(tgt_lengths),
            emotion_text=emotion_text,
            emotion_id=torch.from_numpy(emotion_id),
            cause_ids=torch.from_numpy(cause_ids),
            user_ids=torch.from_numpy(user_ids),
            )
    @property
    def vocab(self):
        r"""The vocabulary, an instance of :class:`~texar.torch.data.Vocab`.
        """
        return self._vocab


class EvalData(tx.data.DatasetBase[Example, Example]):

    def __init__(self, hparams=None,
                 device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        data_source = EvalDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)
        self._vocab = tx.data.Vocab(self._hparams.dataset.vocab_file)
        self._emotion_vocab = EmotionVocab(self._hparams.dataset.emotion_file)
     
        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DatasetBase.default_hparams(),
            'dataset': { 'files': 'data.txt',
                        'compression_type':None, 
                        'vocab_file':None,
                        'emotion_file':None},
        }

    def process(self, raw_example):
        
        utters = raw_example['utters']
        emotion = raw_example['emotion']
        emotion_cause = raw_example['emotion_cause']
        src = []
        cause_ids = [1]
        for idx,u in enumerate(utters[:-1]):
            if idx in emotion_cause:
                cause_ids += [2] * (len(u)+1)
            else:
                cause_ids += [1] * (len(u)+1)
            src.append(' '.join(u + ['<SEP>']))
        src = [' '.join(u + ['<SEP>']) for u in utters[:-1]]
        src = ' '.join(['<CLS>'] + src)
        tgt = ' '.join(['<BOS>']+utters[-1]+['<EOS>'])
        return {
            "src_text": src,
            "src_ids": self._vocab.map_tokens_to_ids_py(src.split(' ')),
            "tgt_text": tgt,
            "tgt_ids": self._vocab.map_tokens_to_ids_py(tgt.split(' ')),
            "emotion_text": emotion,
            "emotion_id": [self._emotion_vocab.token_to_id_map_py[emotion]],
            "cause_ids": np.array(cause_ids)
        }

    def collate(self, examples: List[Example]) -> tx.data.Batch:
        src_text = [ex["src_text"] for ex in examples]
        src_ids, src_lengths = tx.data.padded_batch(
            [ex["src_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)
        tgt_text = [ex["tgt_text"] for ex in examples]
        tgt_ids, tgt_lengths = tx.data.padded_batch(
            [ex["tgt_ids"] for ex in examples], pad_value=self._vocab.pad_token_id)

        cause_ids, cause_lengths = tx.data.padded_batch(
            [ex["cause_ids"] for ex in examples], pad_value=0.0)


        emotion_text = [ex["emotion_text"] for ex in examples]
        emotion_id = np.array([ex["emotion_id"] for ex in examples])
        return tx.data.Batch(
            len(examples),
            src_text=src_text,
            src_text_ids=torch.from_numpy(src_ids),
            src_lengths=torch.tensor(src_lengths),
            tgt_text=tgt_text,
            tgt_text_ids=torch.from_numpy(tgt_ids),
            tgt_lengths=torch.tensor(tgt_lengths),
            emotion_text=emotion_text,
            emotion_id=torch.from_numpy(emotion_id),
            cause_ids=torch.from_numpy(cause_ids),
            )
    @property
    def vocab(self):
        r"""The vocabulary, an instance of :class:`~texar.torch.data.Vocab`.
        """
        return self._vocab


if __name__ == "__main__":
    hparams={
        'dataset': { 'files': 'train.json','vocab_file':'vocab.txt',},
        'batch_size': 10,
        # 'lazy_strategy': 'all',
        # 'num_parallel_calls': 10,
        'shuffle': False
    }
    data = TrainData(hparams)
    iterator = tx.data.DataIterator(data)

    for batch in iterator:
        print(batch)
