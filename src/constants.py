global __version
__version__ = 'v0.1.0'

# common

GPU_DEVICE = 'GPU'
CUDA_DEVICE = 'CUDA_VISIBLE_DEVICES'

# task

TASK_VOCAB = 'vocab'
TAKS_BERT = 'bert'

# for analyzer

LOG_DIR = 'log'
HPARAM_DIR = 'models/hyperparameter'
VOCAB_DIR = 'models/vocab'
PRETRAINING_DIR = 'models/pretraining'
CHECKPOINT_DIR = 'models/checkpoint'
EVAL_DIR = 'models/evaluation'

# model

# for character

NUM_SYMBOL = '<num>'
WORDPIECE_SYMBOL = '##'
SENTENCEPIECE_SYMBOL = '▁'
SENTENCEPIECE_SEPARATOR = '\t'
CTRL_SYMBOL = ['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]']
UNUSED_SYMBOL = 'UNUSED'
NEWLINE_SYMBOL = '\n'

# for data i/o

JSON_FORMAT = 'json'
TXT_FORMAT = 'txt'
ATTR_INFO_DELIM = ','
PREFIX_FIELD_NAME = '_'
FIRST_ELEM = 0
LAST_ELEM = -1
TXT_FORMAT_EXTENSION = '.txt'
JSON_FORMAT_EXTENSION = '.json'
VOCAB_FORMAT_EXTENSION = '.vocab'
TFRECORD_FORMAT_EXTENSION = '.tfrecord'

JSON_SOURCE_KEY = 'source'
JSON_TARGET_KEY = 'target'

# for pretrainer

INSTANCE_DEMO_SIZE = 0
