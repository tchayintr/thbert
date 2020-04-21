import sys

import common
import constants
from data_loaders.data_loader import DataLoader, Data



class CreationDataLoader(DataLoader):
    def __init__(
        self,
        lowercasing=False,
        normalize_digits=True,
    ):
        self.lowercasing = lowercasing
        self.normalize_digits = normalize_digits


    def load_gold_data(self, path, data_format, train=True):
        if data_format == constants.TXT_FORMAT:
            data = self.load_gold_data_TXT(path, train)

        elif data_format == constants.JSON_FORMAT:
            data = self.load_gold_data_JSON(path, train)

        else:
            print('Error: incorrect data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()

        return data


    def load_gold_data_TXT(self, path, train=True):
        data = common.get_lines(path)

        return data


    def load_gold_data_JSON(self, path, train=True):
        sources = []
        targets = []

        with open(path) as f:
            for line in f:
                data = common.get_jsons(line)
                sources.append(common.get_dict_by_indexes(data, constants.JSON_SOURCE_KEY))
                targets.append(common.get_dict_by_indexes(data, constants.JSON_TARGET_KEY))

        if len(sources) != len(targets):
            print('Error: incorrect data length: {}'.format(path), file=sys.stderr)
            print('Error: source length: {}, target length: {}'.format(len(sources), len(targets)), file=sys.stderr)

        return Data(sources, targets)


    def load_gold_dic(self, path, data_format, train=True):
        if data_format == constants.TXT_FORMAT:
            dic = self.load_gold_dic_TXT(path, train)
        elif data_format == constants.JSON_FORMAT:
            dic = None
        else:
            print('Error: incorrect data format: {}'.format(data_format), file=sys.stderr)
            sys.exit()

        return dic


    def load_gold_dic_TXT(self, path, train=True):
        dic = []

        with open(path, 'rt', encoding='utf8') as f:
            for line in f:
                vocab = self.get_sentencepiece_vocab(line)
                if len(vocab) > 0:
                    dic.append(vocab)

        return dic


    def load_gold_dic_JSON(self, path, trian=True):
        pass


    def get_sentencepiece_vocab(self, line):
        vocab = line.split(constants.SENTENCEPIECE_SEPARATOR)[constants.FIRST_ELEM]
        return self.preprocess_token(vocab)


    def gen_vocabs(self, data):
        '''Generating BERT-compatible vocabulary'''
        tokens = data
        token_size = len(data)

        vocabs = []

        for token in tokens:
            vocabs.append(parse_sentencepiece_token(token))

        # skips <unk> symbol then appends bert control symbols in front of the vocabularies
        vocabs = vocabs[1:]
        vocabs = constants.CTRL_SYMBOL + vocabs
        vocabs += ["[{}_{}]".format(constants.UNUSED_SYMBOL, i) for i in range(token_size - len(vocabs))]

        return vocabs



def parse_sentencepiece_token(token):
    if token.startswith(constants.SENTENCEPIECE_SYMBOL) and len(token) > 1:
        return token[1:]
    if token.startswith(constants.SENTENCEPIECE_SYMBOL) and len(token) == 1:
        return constants.WORDPIECE_SYMBOL + token[1:]
    else:
        return constants.WORDPIECE_SYMBOL + token
