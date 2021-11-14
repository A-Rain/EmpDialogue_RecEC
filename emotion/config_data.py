from pathlib import Path

max_batch_tokens = 4096*2
test_batch_size = 32

max_train_epoch = 25
display_steps = 100
eval_steps = 100

max_decoding_length = 1024
data_root = '../Data'
vocab_file = './vocab.txt'
emotion_file = './emotion_set.txt'
glove_file = '/home/lyuhan/datasets/glove/glove.6B.300d.txt'

train_hparams = {
        'dataset': { 'files': data_root + '/train_with_cause.json','vocab_file': vocab_file, "emotion_file": emotion_file},
        'batch_size': 64,
        # 'lazy_strategy': 'all',
        # 'num_parallel_calls': 10,
        'shuffle': True
    }

valid_hparams = {
        'dataset': { 'files': data_root + '/valid_with_cause.json','vocab_file': vocab_file, "emotion_file": emotion_file},
        'batch_size': 64,
        'shuffle': False,
    }

test_hparams = {
        'dataset': { 'files': data_root + '/test_with_cause.json','vocab_file': vocab_file, "emotion_file": emotion_file},
        'batch_size': 64,
        'shuffle': False,
    }