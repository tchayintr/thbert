import json
import pathlib
import sys



def get_dict_by_indexes(dic, indexes):
    return {index: dic[index] for index in indexes}


def get_jsons(fd):
    # str to dictionary (json-like)
    ret = json.loads(fd)
    return ret


def get_json(fd):
    # json to dictionary
    ret = json.load(fd)
    return ret


def get_lines(path):
    with open(path, 'rt') as f:
        return f.readlines()


def get_bert_config_dict(path):
    with open(path, 'rt') as f:
        return json.load(f)


def get_concat_strings(x, y):
    assert isinstance(x, str) and isinstance(y, str)
    ret = x + y
    return ret


def has_attr(var, attr):
    return hasattr(var, attr)


def is_pathlib(var):
    if isinstance(var, pathlib.PosixPath):
        return True
    return False


def report(message, reporter):
    print(message, file=reporter)


def log(message='', logger=sys.stderr):
    print(message, file=logger)
