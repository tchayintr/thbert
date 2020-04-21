import argparse
from pathlib import Path
import sys

import constants



class ArgumentLoader(object):
    def __init__(self):
        # self.parser = argparse.ArgumentParser()
        pass


    def parse_args(self):
        all_args = self.get_full_parser().parse_args()
        min_args = self.get_minimum_parser(all_args).parse_args()
        return min_args


    def get_full_parser(self):
        parser = argparse.ArgumentParser()

        # mode options
        parser.add_argument('--execute_mode', '-x', choices=['train', 'eval', 'hybrid'], help='Choose a mode from among \'train\', \'eval\', and \'hybrid\'')
        parser.add_argument('--quiet', '-q', action='store_true', help='Do not output log file and serialized model file')

        # gpu options
        parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device id (use CPU if specify a negative value)')

        # training parameters
        parser.add_argument('--batch_size', type=int, default=32, help='The number of examples in each mini-batch for training or evaluation (Default: 32)')
        parser.add_argument('--train_batch_size', type=int, default=32, help='The number of examples in each mini-batch for training (Default: 32)')
        parser.add_argument('--eval_batch_size', type=int, default=32, help='The number of examples in each mini-batch for evaluation (Default: 32)')
        parser.add_argument('--checkpoints_steps', type=int, default=2500, help='The number of steps for saving checkpoints (Default: 2500)')
        parser.add_argument('--summary_steps', type=int, default=100, help='The number of steps for saving a summary (Default: 100)')

        # pre-training parameters
        parser.add_argument('--mask_whole_word', action='store_true', help='Mask whole word, otherwise per-WordPiece')
        parser.add_argument('--max_sequence_length', type=int, default=64, help='The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter will be padded (max_seq_length) (Default: 64)')
        parser.add_argument('--max_predictions', type=int, default=20, help='The maximum number of masked Language Model predictions per sequence (max_predictions_per_seq)(Default: 20)')
        parser.add_argument('--masked_lm_prob', type=float, default=0.15, help='Masked Language Model probability (Default: 0.15)')
        parser.add_argument('--random_seed', type=int, default=12345, help='Random seed for pre-training data generation (Default: 12345)')
        parser.add_argument('--dupe_factor', type=int, default=10, help='Number of times to duplicate the input data (with different masks) (Default: 10)')
        parser.add_argument('--short_seq_prob', type=float, default=0.1, help='Probability of creating sequences which are shorter than the maximum length (Default: 0.1)')

        # opimizer parameters
        parser.add_argument('--learning_rate', type=float, default=5e-5, help='Initial learning rate for Adam (Default: 5e-5)')

        # data paths and related options
        parser.add_argument('--hparams_path', type=Path, default=None, help='File path of the predefined hyperparameters')
        parser.add_argument('--checkpoint_path', type=Path, default=None, help='File path of the checkpoint file')
        parser.add_argument('--tfrecord_path', type=Path, default=constants.PRETRAINING_DIR, help='File path of TF examples record (Default: {})'.format(constants.PRETRAINING_DIR))
        parser.add_argument('--input_data_path_prefix', '-p', type=Path, dest='path_prefix', default='.', help='Path prefix of input data (Default: .)')
        parser.add_argument('--train_data', type=Path, default=None, help='File path succeeding \'input_data_path_prefix\' of training data')
        parser.add_argument('--valid_data', type=Path, default=None, help='File path succeeding \'input_data_path_prefix\' of validation data')
        parser.add_argument('--test_data', type=Path, default=None, help='File path succeeding \'input_data_path_prefix\' of testing data')
        parser.add_argument('--dic_data', type=Path, default=None, help='File path succeeding \'input_data_path_prefix\' of input dictionary data (raw vocabularies)')
        parser.add_argument('--vocab_model', type=Path, default=None, help='File path succedding \'input_data_path_prefix\' of SentencePiece vocabulary model')
        parser.add_argument('--input_data_format', type=str.lower, default='txt', help='Choose format of input data among from \'txt\' and \'json\' (Default: txt)')
        parser.add_argument('--dic_data_format', type=str.lower, default='txt', help='Choose format of input dictionary data (raw vocabularies) among from \'txt\' and \'json\' (Default: txt)')
        parser.add_argument('--vocab_path', type=Path, default=constants.VOCAB_DIR, help='File path succedding of WordPiece vocabulary data [will be obtained from --dic_data preprocessing] (Default: {})'.format(constants.VOCAB_DIR))
        parser.add_argument('--eval_result_path', type=Path, default=constants.EVAL_DIR, help='File path of output evaluation result (Default: {})'.format(constants.EVAL_DIR))
        parser.add_argument('--output_model_path', type=Path, default=constants.CHECKPOINT_DIR, help='File path of output model data (Default: {})'.format(constants.CHECKPOINT_DIR))

        # options for data pre/post-processing
        parser.add_argument('--lowercase_alphabets', dest='lowercasing', action='store_true', help='Lowercase alphabets in input text (do_lower_case)')
        parser.add_argument('--normalize_digits', action='store_true', help='Normalize digits by the same symbol in input text')

        return parser


    def get_minimum_parser(self, args):
        parser = argparse.ArgumentParser()

        # basic options
        self.add_basic_options(parser, args)

        # dependent options
        if args.execute_mode == 'train':
            self.add_train_mode_options(parser, args)
        elif args.execute_mode == 'eval':
            self.add_eval_mode_options(parser, args)
        elif args.execute_mode == 'hybrid':
            self.add_hybrid_mode_options(parser, args)
        else:
            print('Error: invalid execute mode: {}'.format(args.execute_mode), file=sys.stderr)
            sys.exit()
        return parser


    def add_basic_options(self, parser, args):
        # mode options
        parser.add_argument('--execute_mode', '-x', required=True, default=args.execute_mode)
        parser.add_argument('--quiet', '-q', action='store_true', default=args.quiet)

        # options for data pre/post-processing
        parser.add_argument('--lowercase_alphabets', dest='lowercasing', action='store_true', default=args.lowercasing)
        parser.add_argument('--normalize_digits', action='store_true', default=args.normalize_digits)

        # gpu options
        parser.add_argument('--gpu', '-g', type=int, default=args.gpu)


    def add_train_mode_options(self, parser, args):
        # training parameters
        parser.add_argument('--batch_size', type=int, default=args.batch_size)
        parser.add_argument('--checkpoints_steps', type=int, default=args.checkpoints_steps)
        parser.add_argument('--summary_steps', type=int, default=args.summary_steps)

        # pre-training parameters
        parser.add_argument('--mask_whole_word', action='store_true', default=args.mask_whole_word)
        parser.add_argument('--max_sequence_length', type=int, default=args.max_sequence_length)
        parser.add_argument('--max_predictions', type=int, default=args.max_predictions)
        parser.add_argument('--random_seed', type=int, default=args.random_seed)
        parser.add_argument('--dupe_factor', type=int, default=args.dupe_factor)
        parser.add_argument('--masked_lm_prob', type=float, default=args.masked_lm_prob)
        parser.add_argument('--short_seq_prob', type=float, default=args.short_seq_prob)

        # data paths and related options
        parser.add_argument('--hparams_path', default=args.hparams_path)
        parser.add_argument('--checkpoint_path', default=args.checkpoint_path)
        parser.add_argument('--tfrecord_path', type=Path, default=args.tfrecord_path)
        parser.add_argument('--input_data_path_prefix', '-p', type=Path, dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--train_data', type=Path, default=args.train_data)
        parser.add_argument('--valid_data', type=Path, default=args.valid_data)
        parser.add_argument('--dic_data', type=Path, required=True, default=args.dic_data)
        parser.add_argument('--vocab_model', type=Path, required=True, default=args.vocab_model)
        parser.add_argument('--vocab_path', type=Path, default=args.vocab_path)
        parser.add_argument('--eval_result_path', type=Path, default=args.eval_result_path)
        parser.add_argument('--output_model_path', default=args.output_model_path)
        self.add_input_data_format_option(parser, args)
        self.add_dic_data_format_option(parser, args)

        # model parameters

        # optimizer parameters
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)


    def add_eval_mode_options(self, parser, args):
        # evaluation parameters
        parser.add_argument('--batch_size', type=int, default=args.batch_size)
        parser.add_argument('--checkpoints_steps', type=int, default=args.checkpoints_steps)
        parser.add_argument('--summary_steps', type=int, default=args.summary_steps)

        # pre-training parameters
        parser.add_argument('--mask_whole_word', action='store_true', default=args.mask_whole_word)
        parser.add_argument('--max_sequence_length', type=int, default=args.max_sequence_length)
        parser.add_argument('--max_predictions', type=int, default=args.max_predictions)
        parser.add_argument('--random_seed', type=int, default=args.random_seed)
        parser.add_argument('--dupe_factor', type=int, default=args.dupe_factor)
        parser.add_argument('--masked_lm_prob', type=float, default=args.masked_lm_prob)
        parser.add_argument('--short_seq_prob', type=float, default=args.short_seq_prob)

        # data paths and related options
        parser.add_argument('--hparams_path', default=args.hparams_path)
        parser.add_argument('--checkpoint_path', default=args.checkpoint_path)
        parser.add_argument('--tfrecord_path', type=Path, default=args.tfrecord_path)
        parser.add_argument('--input_data_path_prefix', '-p', type=Path, dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--train_data', type=Path, default=args.train_data)
        parser.add_argument('--valid_data', type=Path, default=args.valid_data)
        parser.add_argument('--dic_data', type=Path, required=True, default=args.dic_data)
        parser.add_argument('--vocab_model', type=Path, required=True, default=args.vocab_model)
        parser.add_argument('--vocab_path', type=Path, default=args.vocab_path)
        parser.add_argument('--eval_result_path', type=Path, default=args.eval_result_path)
        parser.add_argument('--output_model_path', default=args.output_model_path)
        self.add_input_data_format_option(parser, args)
        self.add_dic_data_format_option(parser, args)

        # model parameters

        # optimizer parameters
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)


    def add_hybrid_mode_options(self, parser, args):
        # training parameters
        parser.add_argument('--batch_size', type=int, default=args.batch_size)
        parser.add_argument('--checkpoints_steps', type=int, default=args.checkpoints_steps)
        parser.add_argument('--summary_steps', type=int, default=args.summary_steps)


        # pre-training parameters
        parser.add_argument('--mask_whole_word', action='store_true', default=args.mask_whole_word)
        parser.add_argument('--max_sequence_length', type=int, default=args.max_sequence_length)
        parser.add_argument('--max_predictions', type=int, default=args.max_predictions)
        parser.add_argument('--random_seed', type=int, default=args.random_seed)
        parser.add_argument('--dupe_factor', type=int, default=args.dupe_factor)
        parser.add_argument('--masked_lm_prob', type=float, default=args.masked_lm_prob)
        parser.add_argument('--short_seq_prob', type=float, default=args.short_seq_prob)

        # data paths and related options
        parser.add_argument('--hparams_path', default=args.hparams_path)
        parser.add_argument('--checkpoint_path', default=args.checkpoint_path)
        parser.add_argument('--tfrecord_path', type=Path, default=args.tfrecord_path)
        parser.add_argument('--input_data_path_prefix', '-p', type=Path, dest='path_prefix', default=args.path_prefix)
        parser.add_argument('--train_data', type=Path, default=args.train_data)
        parser.add_argument('--valid_data', type=Path, default=args.valid_data)
        parser.add_argument('--dic_data', type=Path, required=True, default=args.dic_data)
        parser.add_argument('--vocab_model', type=Path, required=True, default=args.vocab_model)
        parser.add_argument('--vocab_path', type=Path, default=args.vocab_path)
        parser.add_argument('--eval_result_path', type=Path, default=args.eval_result_path)
        parser.add_argument('--output_model_path', default=args.output_model_path)
        self.add_input_data_format_option(parser, args)
        self.add_dic_data_format_option(parser, args)

        # model parameters

        # optimizer parameters
        parser.add_argument('--learning_rate', type=float, default=args.learning_rate)


    def add_input_data_format_option(self, parser, args):
        parser.add_argument('--input_data_format', default=args.input_data_format)


    def add_dic_data_format_option(self, parser, args):
        parser.add_argument('--dic_data_format', default=args.dic_data_format)



if __name__ == '__main__':
    ArgumentLoader().get_full_parser()
