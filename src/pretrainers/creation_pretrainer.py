'''Create masked LM/next sentence masked_lm TF examples for BERT'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import random
import tensorflow as tf
import sys

import common
import constants
from pretrainers import pretrainer
from pretrainers.pretrainer import Pretrainer
import tokenization



class PretrainingInstance(object):
    '''A single pre-training instance (sentence pair)'''
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


    def __str__(self):
        s = ''
        s += 'tokens: {}\n'.format(' '.join([tokenization.printable_text(x) for x in self.tokens]))
        s += 'segment_ids: {}\n'.format(' '.join([str(x) for x in self.segment_ids]))
        s += 'is_random_next: {}\n'.format(self.is_random_next)
        s += 'masked_lm_positions: {}\n'.format(' '.join([str(x) for x in self.masked_lm_positions]))
        s += 'masked_lm_labels: {}\n'.format(' '.join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += '\n'
        return s


    def __repr__(self):
        return self.__str__()



class CreationPretrainer(Pretrainer):
    def __init__(
        self,
        input_train_file,
        input_valid_file,
        output_record_path,
        vocab_data,
        do_lower_case,
        do_whole_word,
        max_seq_length,
        max_predictions_per_seq,
        masked_lm_prob,
        random_seed,
        dupe_factor,
        short_seq_prob,
        attention_probs_dropout_prob,
        bert_config_path,
        directionality,
        hidden_act,
        hidden_dropout_prob,
        hidden_size,
        initializer_range,
        intermediate_size,
        max_position_embeddings,
        num_attention_heads,
        num_hidden_layers,
        pooler_fc_size,
        pooler_num_attention_heads,
        pooler_num_fc_layers,
        pooler_size_per_head,
        pooler_type,
        type_vocab_size,
        vocab_size,
        batch_size,
        checkpoint_path,
        eval_result_path,
        learning_rate,
        max_eval_steps,
        model_path,
        num_train_steps,
        num_warmup_steps,
        save_checkpoints_steps,
        save_summary_steps,
        use_one_hot_embeddings,
        use_tpu,
        execute_mode,
        quiet,
        logger,
        reporter,
    ):
        super().__init__(
            input_train_file,
            input_valid_file,
            output_record_path,
            vocab_data,
            do_lower_case,
            do_whole_word,
            max_seq_length,
            max_predictions_per_seq,
            masked_lm_prob,
            random_seed,
            dupe_factor,
            short_seq_prob,
            attention_probs_dropout_prob,
            bert_config_path,
            directionality,
            hidden_act,
            hidden_dropout_prob,
            hidden_size,
            initializer_range,
            intermediate_size,
            max_position_embeddings,
            num_attention_heads,
            num_hidden_layers,
            pooler_fc_size,
            pooler_num_attention_heads,
            pooler_num_fc_layers,
            pooler_size_per_head,
            pooler_type,
            type_vocab_size,
            vocab_size,
            batch_size,
            checkpoint_path,
            eval_result_path,
            learning_rate,
            max_eval_steps,
            model_path,
            num_train_steps,
            num_warmup_steps,
            save_checkpoints_steps,
            save_summary_steps,
            use_one_hot_embeddings,
            use_tpu,
            execute_mode,
            quiet,
            logger,
            reporter,
        )


    def gen_pretraining_data(self):
        '''Generate TF examples record to output file'''
        self.log('Generate pretraining data')
        self.init_data_path()
        self.init_io_data()
        self.setup_tokenizer()
        self.setup_random_generator()
        self.load_input_data()
        self.gen_data_instances()
        self.gen_data_record()


    def gen_data_record(self):
        if self.execute_mode == 'train':
            self.gen_record('train')
        elif self.execute_mode == 'eval':
            self.gen_record('valid')
        elif self.execute_mode == 'hybrid':
            self.gen_record('train')
            self.gen_record('valid')


    def gen_record(self, data_type):
        record_outputs = None
        if data_type == 'train':
            self.tfrecord_train_data = write_instance_to_example_files(
                self.train_instances,
                self.tokenizer,
                self.max_seq_length,
                self.max_predictions_per_seq,
                self.record_train_outputs,
                self.log,
                self.report
            )
            record_outputs = self.record_train_outputs

        elif data_type == 'valid':
            self.tfrecord_valid_data = write_instance_to_example_files(
                self.valid_instances,
                self.tokenizer,
                self.max_seq_length,
                self.max_predictions_per_seq,
                self.record_valid_outputs,
                self.log,
                self.report
            )
            record_outputs = self.record_valid_outputs

        self.log('### Dump output {} tfrecord'.format(data_type))
        for _output in record_outputs:
            self.log('# {}'.format(_output))
        self.log()


    def gen_data_instances(self):
        if self.execute_mode == 'train':
            self.train_instances = self.gen_instances('train')
        elif self.execute_mode == 'eval':
            self.valid_instances = self.gen_instances('valid')
        elif self.execute_mode == 'hybrid':
            self.train_instances = self.gen_instances('train')
            self.valid_instances = self.gen_instances('valid')


    def gen_instances(self, data_type):
        if data_type == 'train':
            input_data = self.train_data
        elif data_type == 'valid':
            input_data = self.valid_data

        instances = create_training_instances(
            input_data,
            self.tokenizer,
            self.max_seq_length,
            self.dupe_factor,
            self.short_seq_prob,
            self.masked_lm_prob,
            self.max_predictions_per_seq,
            self.do_whole_word,
            self.random_generator
        )
        return instances


    def init_data_path(self):
        if self.execute_mode == 'train':
            self.init_path('train')
        elif self.execute_mode == 'eval':
            self.init_path('valid')
            self.init_path('eval')
        elif self.execute_mode == 'hybrid':
            self.init_path('train')
            self.init_path('valid')
            self.init_path('eval')

        self.init_path('record')


    def init_path(self, path_type):
        if path_type == 'train' and common.is_pathlib(self.input_train_file):
            self.input_train_file = self.input_train_file.as_posix()
            ext = '{}{}{}'.format(
                self.init_time,
                '-train' if self.execute_mode == 'hybrid' else '',
                constants.TFRECORD_FORMAT_EXTENSION
            )
            tfrecord_train_path = self.output_record_path / ext
            self.tfrecord_train_path = tfrecord_train_path.as_posix()

        elif path_type == 'valid' and common.is_pathlib(self.input_valid_file):
            self.input_valid_file = self.input_valid_file.as_posix()
            ext = '{}{}{}'.format(
                self.init_time,
                '-valid' if self.execute_mode == 'hybrid' else '',
                constants.TFRECORD_FORMAT_EXTENSION
            )
            tfrecord_valid_path = self.output_record_path / ext
            self.tfrecord_valid_path = tfrecord_valid_path.as_posix()

        elif path_type == 'eval' and self.eval_result_path:
            eval_summary_path = self.eval_result_path / (self.init_time + constants.TXT_FORMAT_EXTENSION)
            self.eval_summary_path = eval_summary_path.as_posix()

        # elif path_type == 'record' and common.is_pathlib(self.output_record_path):
        #     output_record_path = self.output_record_path / (self.init_time + constants.TFRECORD_FORMAT_EXTENSION)
        #     self.output_record_path = output_record_path.as_posix()


    def init_io_data(self):
        if self.execute_mode == 'train':
            self.init_data('train')
        elif self.execute_mode == 'eval':
            self.init_data('valid')
        elif self.execute_mode == 'hybrid':
            self.init_data('train')
            self.init_data('valid')

        self.init_data('output')


    def init_data(self, data_type):
        if data_type == 'train':
            self.train_inputs = self.input_train_file.split(',')

        elif data_type == 'valid':
            self.valid_inputs = self.input_valid_file.split(',')

        elif data_type == 'output':
            self.record_train_outputs = self.tfrecord_train_path.split(',') if self.tfrecord_train_path else None
            self.record_valid_outputs = self.tfrecord_valid_path.split(',') if self.tfrecord_valid_path else None

        else:
            print('Error: incorrect data type: {}'.format(data_type), file=sys.stderr)
            sys.exit()


    def init_bert_configurations(self):
        bert_configs = {
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'directionality': self.directionality,
            'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'hidden_size': self.hidden_size,
            'initializer_range': self.initializer_range,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'pooler_fc_size': self.pooler_fc_size,
            'pooler_num_attention_heads': self.pooler_num_attention_heads,
            'pooler_num_fc_layers': self.pooler_num_fc_layers,
            'pooler_size_per_head': self.pooler_size_per_head,
            'pooler_type': self.pooler_type,
            'type_vocab_size': self.type_vocab_size,
            'vocab_size': self.vocab_size,
        }

        self.log('Init BERT configurations')
        self.log('# arguments')
        for k, v in bert_configs.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] bert-config: {}'.format(message))
        self.log('')
        self.bert_configs = pretrainer.modeling.BertConfig.from_dict(bert_configs)

        if not self.quiet:
            output_bert_config = '{}/{}-bert_config.json'.format(constants.HPARAM_DIR, self.init_time)
            with open(output_bert_config, 'w') as f:
                json.dump(bert_configs, f)


    def init_train_checkpoint(self):
        self.init_checkpoint = tf.train.latest_checkpoint(self.model_path)


    def init_run_configurations(self):
        self.run_configs = tf.estimator.RunConfig(
            model_dir=self.model_path,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoints_steps
        )


    def init_model_fn(self):
        self.init_train_checkpoint()
        self.model_fn = pretrainer.model_fn_builder(
            bert_config=self.bert_configs,
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )


    def init_train_input_fn(self):
        self.train_input_fn = pretrainer.input_fn_builder(
            input_files=self.record_train_outputs,
            max_seq_length=self.max_seq_length,
            max_predictions_per_seq=self.max_predictions_per_seq,
            is_training=True,
        )


    def init_eval_input_fn(self):
        self.eval_input_fn = pretrainer.input_fn_builder(
            input_files=self.record_valid_outputs,
            max_seq_length=self.max_seq_length,
            max_predictions_per_seq=self.max_predictions_per_seq,
            is_training=False,
        )


    def init_estimator_hyperparameters(self):
        estimator_hparams = {}
        estimator_hparams['batch_size'] = self.batch_size
        self.estimator_hparams = estimator_hparams


    def load_bert_configurations(self):
        self.bert_configs = pretrainer.modeling.BertConfig.from_json_file(self.bert_config_path)

        bert_configs = pretrainer.modeling.BertConfig.to_dict(self.bert_configs)
        self.log('Load BERT configurations')
        self.log('# arguments')
        for k, v in bert_configs.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] bert-config: {}'.format(message))
        self.log('')

        if not self.quiet:
            output_bert_config = '{}/{}.json'.format(constants.HPARAM_DIR, self.init_time)
            with open(output_bert_config, 'w') as f:
                json.dump(bert_configs, f)


    def setup_random_generator(self):
        self.random_generator = random.Random(self.random_seed)


    def setup_tokenizer(self):
        vocab_data_path = self.vocab_data
        if common.is_pathlib(vocab_data_path):
            vocab_data_path = vocab_data_path.as_posix()

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_data_path,
            do_lower_case=self.do_lower_case
        )


    def setup_estimator(self, train=True):
        if self.bert_config_path:
            self.load_bert_configurations()
        else:
            self.init_bert_configurations()
        self.init_run_configurations()
        self.init_model_fn()

        if train:
            self.init_train_input_fn()
        elif not train:
            self.init_eval_input_fn()

        self.init_estimator_hyperparameters()
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=self.run_configs,
            params=self.estimator_hparams,
        )



def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, outputs, log_fn, report_fn):
    '''Create TF example files from `TrainingInstance`s'''
    tf_examples = []
    writers = []
    for _output in outputs:
        writers.append(tf.io.TFRecordWriter(_output))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(input_ids)
        features['input_mask'] = create_int_feature(input_mask)
        features['segment_ids'] = create_int_feature(segment_ids)
        features['masked_lm_positions'] = create_int_feature(masked_lm_positions)
        features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
        features['masked_lm_weights'] = create_float_feature(masked_lm_weights)
        features['next_sentence_labels'] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_examples.append(tf_example)

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        # DEMO
        if inst_index < constants.INSTANCE_DEMO_SIZE:
            log_fn('### Example')
            log_fn('tokens: {}'.format(' '.join([tokenization.printable_text(x) for x in instance.tokens])))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                log_fn('{}: {}'.format(feature_name, ' '.join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    log_fn('### Generate TF example')
    log_fn('# {} instances'.format(total_written))
    report_fn('[INFO] TF example: {} instances'.format(total_written))

    return tf_examples


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_data, tokenizer, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq, do_whole_word, rng):
    '''Create `TrainingInstance`s from raw text'''
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for _input in input_data:
        with tf.io.gfile.GFile(_input, 'r') as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents,
                    document_index,
                    max_seq_length,
                    short_seq_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    do_whole_word,
                    vocab_words,
                    rng
                )
            )

    rng.shuffle(instances)
    return instances


def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, do_whole_word, vocab_words, rng):
    '''Creates `TrainingInstance`s for a single document'''
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append('[CLS]')
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append('[SEP]')
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append('[SEP]')
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    do_whole_word,
                    vocab_words,
                    rng
                )

                instance = PretrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, do_whole_word, vocab_words, rng):
    '''Creates the predictions for the masked LM objective'''
    MaskedLmInstance = collections.namedtuple('MaskedLmInstance', ['index', 'label'])

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word and len(cand_indexes) >= 1 and token.startswith('##')):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    '''Truncates a pair of sequences to a maximum sequence length'''
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
