from datetime import datetime
import numpy as np
import sys

import constants



class Trainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--input_data'
                )
                err_msgs.append(msg)
            if not args.dic_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--dic_data')
                err_msgs.append(msg)
            if not args.vocab_model:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--vocab_model'
                )
                err_msgs.append(msg)
            if not args.vocab_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--vocab_path'
                )
            if not args.output_model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--output_model_path'
                )
                err_msgs.append(msg)

        elif args.execute_mode == 'eval':
            if not args.valid_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--valid_data'
                )
                err_msgs.append(msg)
            if not args.dic_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--dic_data'
                )
            if not args.eval_result_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--eval_result_path'
                )
                err_msgs.append(msg)
            if not args.output_model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--output_model_path'
                )
                err_msgs.append(msg)

        elif args.execute_mode == 'hybrid':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--input_data'
                )
                err_msgs.append(msg)
            if not args.valid_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--valid_data'
                )
                err_msgs.append(msg)
            if not args.dic_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--dic_data')
                err_msgs.append(msg)
            if not args.vocab_model:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--vocab_model'
                )
                err_msgs.append(msg)
            if not args.vocab_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--vocab_path'
                )
            if not args.eval_result_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--eval_result_path'
                )
                err_msgs.append(msg)
            if not args.output_model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--output_model_path'
                )
                err_msgs.append(msg)

        else:
            msg = 'Error: invalid execute mode: {}'.format(args.execute_mode)
            err_msgs.append(msg)

        if err_msgs:
            for msg in err_msgs:
                print(msg, file=sys.stderr)
            sys.exit()

        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = logger    # output execute log
        self.reporter = None    # output evaluation results
        self.train = None
        self.valid = None
        self.test = None
        self.dic = None
        self.vocabs = None
        self.hparams = None
        self.data_loader = None
        self.pretrainer = None

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), mode='a')


    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message=''):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def show_hyperparameters(self):
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if k in self.hparams and v == self.hparams[k]:
                message = '{}={}'.format(k, self.hparams[k])

            elif k in self.hparams:
                if v != self.hparams[k] and (str(v) == str(self.hparams[k])):
                    message = '{}={}'.format(k, self.hparams[k])
                elif v != self.hparams[k]:
                    message = '{}={} (input option value {} was discarded)'.format(k, v, self.hparams[k])
                    self.hparams[k] = v

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def update_model(self, classifier=None, train=False):
        # to be implemented in sub-class
        pass


    def init_hyperparameters(self):
        # to be implemented in sub-class
        self.hparams = {}


    def update_hyperparameter(self, key, value):
        # to be implemented in sub-class
        pass


    def load_hyperparameters(self, hparam_path):
        # to be implemented in sub-class
        pass


    def load_training_and_validation_data(self):
        self.load_data('train')
        if self.args.valid_data:
            self.load_data('valid')
        self.load_data('dic')
        self.show_training_data()


    def load_training_data(self):
        self.load_data('train')
        self.load_data('dic')
        self.show_training_data()


    def load_validation_data(self):
        self.load_data('valid')
        self.load_data('dic')
        self.show_training_data()


    def load_test_data(self):
        self.load_data('test')


    def load_data(self, data_type):
        if data_type == 'train':
            self.setup_data_loader()
            data_path = self.args.path_prefix / self.args.train_data
            data = self.data_loader.load_gold_data(data_path, self.args.input_data_format, train=True)
            self.train = data

        elif data_type == 'valid':
            if not self.train:
                self.setup_data_loader()
            data_path = self.args.path_prefix / self.args.valid_data
            data = self.data_loader.load_gold_data(data_path, self.args.input_data_format, train=False)
            self.valid = data

        elif data_type == 'test':
            self.setup_data_loader()
            data_path = self.args.path_prefix / self.args.test_data
            data = self.data_loader.load_gold_data(data_path, self.args.input_data_format, train=False)
            self.test = data

        elif data_type == 'dic':
            self.setup_data_loader()
            data_path = self.args.path_prefix / self.args.dic_data
            self.dic = self.data_loader.load_gold_dic(data_path, self.args.dic_data_format, train=True)
            self.vocabs = self.gen_vocabs()
            if self.args.vocab_path:
                vocab_data_path = self.args.vocab_path / (self.start_time + constants.VOCAB_FORMAT_EXTENSION)
                dump_vocab_file(self.vocabs, vocab_data_path)
                self.update_hyperparameter(key='vocab_data_path', value=vocab_data_path)
                self.log('Dump vocab data: {}'.format(vocab_data_path))
                self.log()

        else:
            print('Error: incorrect data type: {}'.format(data_type), file=sys.stderr)
            sys.exit()

        self.log('Load {} data: {}'.format(data_type, data_path))
        self.show_data_info(data_type)


    def show_data_info(self):
        # to be implemented in sub-class
        pass


    def show_training_data(self):
        # to be implemented in sub-class
        pass


    def setup_data_loader(self):
        # to be implemented in sub-class
        pass


    def setup_optimizer(self):
        # to be implemented in sub-class
        pass


    def setup_evaluator(self, evaluator=None):
        # to be implemented in sub-class
        pass


    def setup_pretrainer(self):
        # to be implemented in sub-class
        pass


    def gen_vocabs(self):
        if self.dic:
            vocabs = self.data_loader.gen_vocabs(self.dic)
        else:
            print('Error: incorrupt dictionary data'.format(file=sys.stderr))
            sys.exit()
        return vocabs


    def run_train_mode(self):
        self.pretrainer.run_pretrainer('train')
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.HPARAM_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('[INFO] complete: {}\n'.format(time))
        self.log('Finish: {}\n'.format(time))


    def run_eval_mode(self):
        self.pretrainer.run_pretrainer('eval')
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.HPARAM_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('[INFO] complete: {}\n'.format(time))
        self.log('Finish: {}\n'.format(time))


    def run_hybrid_mode(self):
        self.pretrainer.run_pretrainer('hybrid')
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.HPARAM_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('[INFO] complete: {}\n'.format(time))
        self.log('Finish: {}\n'.format(time))


def batch_generator(len_data, batch_size=100, shuffle=True):
    perm = np.random.permutation(len_data) if shuffle else list(range(0, len_data))
    for i in range(0, len_data, batch_size):
        i_max = min(i + batch_size, len_data)
        yield perm[i:i_max]


def dump_vocab_file(vocabs, path):
    with open(path, 'w', encoding='utf8') as f:
        for vocab in vocabs:
            f.write(vocab + constants.NEWLINE_SYMBOL)
