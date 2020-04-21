import re
import pickle

import constants



class Data(object):
    def __init__(self, inputs=None, outputs=None):
        self.inputs = inputs		# list of input sequences e.g. chars, words
        self.outputs = outputs       	# list of output sequences (label sequence)


    @property
    def __len__(self):
        return len(self.inputs) if len(self.inputs) == len(self.outputs) else None



class DataLoader(object):
    def __init__(self):
        self.lowercasing = False
        self.normalize_digits = False


    def load_gold_data(self, path, data_format, train=True):
        # to be implemented in sub-class
        pass


    def load_gold_vocab(self, path, data_format, train=True):
        # to be implemented in sub-class
        pass


    def preprocess_token(self, token):
        if self.lowercasing:
            token = str(token).lower()
        if self.normalize_digits:
            token = re.sub(r'[0-9๐-๙]+', constants.NUM_SYMBOL, token)

        return token


def load_pickled_data(filename_wo_ext):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)

    return Data(obj)


def dump_pickled_data(filename_wo_ext, data):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data)
        obj = pickle.dump(obj, f)
