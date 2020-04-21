'''Run masked LM/next sentence masked_lm pre-training for BERT'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import modeling
import optimization
import tensorflow as tf
import sys



class Pretrainer(object):
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
        # pretraining data generation
        self.input_train_file = input_train_file
        self.input_valid_file = input_valid_file
        self.output_record_path = output_record_path
        self.vocab_data = vocab_data
        self.do_lower_case = do_lower_case
        self.do_whole_word = do_whole_word
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = masked_lm_prob
        self.random_seed = random_seed
        self.dupe_factor = dupe_factor
        self.short_seq_prob = short_seq_prob

        # BERT configs
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bert_config_path = bert_config_path
        self.directionality = directionality
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size

        # estimator

        # common
        self.batch_size = batch_size
        self.use_tpu = use_tpu

        ### model functions
        self.learning_rate = learning_rate
        self.max_eval_steps = max_eval_steps
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_one_hot_embeddings = use_one_hot_embeddings

        ### run configs
        # output_model_data
        self.checkpoint_path = checkpoint_path
        self.eval_result_path = eval_result_path
        self.model_path = model_path
        self.save_checkpoints_steps = save_checkpoints_steps
        self.save_summary_steps = save_summary_steps

        self.execute_mode = execute_mode
        self.quiet = quiet
        self.logger = logger
        self.reporter = reporter

        self.init_time = datetime.now().strftime('%Y%m%d_%H%M')

        self.train_data = None
        self.valid_data = None
        self.inputs = None
        self.train_inputs = None
        self.valid_inputs = None
        self.record_train_outputs = None
        self.record_valid_outputs = None
        self.tokenizer = None
        self.random_generator = None
        self.train_instances = None
        self.valid_instances = None
        self.tfrecord_train_data = None
        self.tfrecord_valid_data = None
        self.tfrecord_train_path = None
        self.tfrecord_valid_path = None
        self.tfrecord_files = None

        self.bert_configs = None
        self.run_configs = None
        self.init_checkpoint = None
        self.model_fn = None
        self.train_input_fn = None
        self.eval_input_fn = None
        self.estimator_hparams = None
        self.estimator = None
        self.eval_result = None
        self.eval_summary_path = None


    def report(self, message):
        if not self.quiet:
            print(message, file=self.reporter)


    def log(self, message=''):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def gen_eval_summary(self):
        eval_result = self.eval_result
        output_eval_summary_path = self.eval_summary_path

        with tf.io.gfile.GFile(output_eval_summary_path, 'w') as f:
            self.log('### Evaluation summary')
            for key in sorted(self.eval_result.keys()):
                self.log('# {} = {}'.format(key, str(eval_result[key])))
                self.report('[SUMMARY] evaluation: {}={}'.format(key, str(eval_result[key])))
                f.write('{}: {}\n'.format(key, str(eval_result[key])))
        self.log('')


    def gen_pretraining_data(self):
        # to be implemented in sub-class
        pass


    def gen_data_record(self):
        # to be implemented in sub-class
        pass


    def gen_record(self, data_type):
        # to be implemented in sub-class
        pass


    def gen_data_instances(self):
        # to be implemented in sub-class
        pass

    def gen_instances(self, data_type):
        # to be implemented in sub-class
        pass


    def init_data_path(self):
        # to be implemented in sub-class
        pass


    def init_path(self):
        # to be implemented in sub-class
        pass


    def init_io_data(self):
        # to be implemented in sub-class
        self.train_inputs = []
        self.valid_inputs = []
        self.record_train_outputs = []
        self.record_valid_outputs = []


    def init_data(self, data_type):
        # to be implemented in sub-class
        pass


    def init_bert_configurations(self):
        # to be implemented in sub-class
        pass


    def init_run_configurations(self):
        # to be implemented in sub-class
        pass


    def init_train_checkpoint(self):
        # to be implemented in sub-class
        pass


    def init_model_fn(self):
        # to be implemented in sub-class
        pass


    def init_train_input_fn(self):
        # to be implemented in sub-class
        pass


    def init_eval_input_fn(self):
        # to be implemented in sub-class
        pass


    def init_estimator_hyperparameters(self):
        # to be implemented in sub-class
        self.estimator_hparams = {}


    def load_bert_configurations(self):
        # to be implemented in sub-class
        pass


    def load_input_data(self):
        if self.execute_mode == 'train':
            self.load_data('train')
        elif self.execute_mode == 'eval':
            self.load_data('valid')
        elif self.execute_mode == 'hybrid':
            self.load_data('train')
            self.load_data('valid')



    def load_data(self, data_type):
        input_data = []
        if data_type == 'train':
            for input_pattern in self.input_train_file.split(','):
                input_data.extend(tf.io.gfile.glob(input_pattern))
            self.train_data = input_data

        elif data_type == 'valid':
            for input_pattern in self.input_valid_file.split(','):
                input_data.extend(tf.io.gfile.glob(input_pattern))
            self.valid_data = input_data

        else:
            print('Error: incorrect data type: {}'.format(data_type), file=sys.stderr)
            sys.exit()

        self.log('### Load {} data'.format(data_type))
        for _input in input_data:
            self.log('# {}'.format(_input))


    def setup_random_generator(self):
        # to be implemented in sub-class
        pass


    def setup_tokenizer(self):
        # to be implemented in sub-class
        pass


    def setup_estimator(self):
        # to be implemented in sub-class
        pass


    def run_pretrainer(self, mode):
        if mode == 'train':
            self.log('### Running training')
            self.log('# batch size: {}\n'.format(self.batch_size))
            self.setup_estimator(train=True)
            self.estimator.train(
                input_fn=self.train_input_fn,
                max_steps=self.num_train_steps
            )

        elif mode == 'eval':
            self.log('### Running evaluation')
            self.log('# batch size: {}\n'.format(self.batch_size))
            self.setup_estimator(train=False)
            self.eval_result = self.estimator.evaluate(
                input_fn=self.eval_input_fn,
                steps=self.max_eval_steps
            )
            self.gen_eval_summary()


        elif mode == 'hybrid':
            self.log('### Running training')
            self.log('# batch size: {}\n'.format(self.batch_size))
            self.setup_estimator(train=True)
            self.estimator.train(
                input_fn=self.train_input_fn,
                max_steps=self.num_train_steps
            )

            self.log('### Running evaluation')
            self.log('# batch size: {}\n'.format(self.batch_size))
            self.setup_estimator(train=False)
            self.eval_result = self.estimator.evaluate(
                input_fn=self.eval_input_fn,
                steps=self.max_eval_steps
            )
            self.gen_eval_summary()



def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    '''Returns `model_fn` closure for TPUEstimator'''

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        '''The `model_fn` for TPUEstimator'''

        # tf.logging.info('*** Features ***')
        # for name in sorted(features.keys()):
        #     tf.logging.info('  name = {}, shape = {}'.format(name, features[name].shape))

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        masked_lm_positions = features['masked_lm_positions']
        masked_lm_ids = features['masked_lm_ids']
        masked_lm_weights = features['masked_lm_weights']
        next_sentence_labels = features['next_sentence_labels']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(),
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights
        )

        (next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output(
            bert_config,
            model.get_pooled_output(),
            next_sentence_labels
        )

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars,
                init_checkpoint
            )
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info('**** Trainable Variables ****')
        print('### Trainable variables')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            print('# name: {}, shape: {}{},'.format(var.name, var.shape, init_string))
            # tf.logging.info('  name = {}, shape = {}{}'.format(var.name, var.shape, init_string))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss,
                learning_rate,
                num_train_steps,
                num_warmup_steps,
                use_tpu
            )

            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                # scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, next_sentence_example_loss, next_sentence_log_probs, next_sentence_labels):
                '''Computes the loss and accuracy of the model'''
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                # masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                #     labels=masked_lm_ids,
                #     predictions=masked_lm_predictions,
                #     weights=masked_lm_weights
                # )
                masked_lm_accuracy = tf.metrics.Accuracy()
                masked_lm_accuracy.update_state(
                    y_true=masked_lm_ids,
                    y_pred=masked_lm_predictions,
                    sample_weight=masked_lm_weights
                )

                # masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                #     values=masked_lm_example_loss,
                #     weights=masked_lm_weights
                # )
                masked_lm_mean_loss = tf.metrics.Mean()
                masked_lm_mean_loss.update_state(
                    values=masked_lm_example_loss,
                    sample_weight=masked_lm_weights
                )

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs,
                    [-1, next_sentence_log_probs.shape[-1]]
                )
                next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                # next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                #     labels=next_sentence_labels,
                #     predictions=next_sentence_predictions
                # )
                next_sentence_accuracy = tf.metrics.Accuracy()
                next_sentence_accuracy.update_state(
                    y_true=next_sentence_labels,
                    y_pred=next_sentence_predictions
                )
                # next_sentence_mean_loss = tf.metrics.mean(values=next_sentence_example_loss)
                next_sentence_mean_loss = tf.metrics.Mean()
                next_sentence_mean_loss.update_state(values=next_sentence_example_loss)

                return {
                    'masked_lm_accuracy': masked_lm_accuracy,
                    'masked_lm_loss': masked_lm_mean_loss,
                    'next_sentence_accuracy': next_sentence_accuracy,
                    'next_sentence_loss': next_sentence_mean_loss,
                }

            # eval_metrics = (metric_fn, [
            #     masked_lm_example_loss,
            #     masked_lm_log_probs,
            #     masked_lm_ids,
            #     masked_lm_weights,
            #     next_sentence_example_loss,
            #     next_sentence_log_probs,
            #     next_sentence_labels
            # ])
            eval_metrics = metric_fn(
                masked_lm_example_loss,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_example_loss,
                next_sentence_log_probs,
                next_sentence_labels
            )

            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                # scaffold_fn=scaffold_fn
            )

        else:
            raise ValueError('Only TRAIN and EVAL modes are supported: {}'.format(mode))

        return output_spec

    return model_fn


# TODO
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
    '''Get loss and log probs for the masked LM'''
    input_tensor = gather_indexes(input_tensor, positions)

    # with tf.variable_scope('cls/predictions'):
    with tf.name_scope('cls/predictions'):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        # with tf.variable_scope('transform'):
        with tf.name_scope('transform'):
            # input_tensor = tf.layers.dense(
            input_tensor = tf.keras.layers.Dense(
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range)
            )(input_tensor)
            input_tensor = modeling.layer_norm(input_tensor)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # output_bias = tf.get_variable(
        #     'output_bias',
        #     shape=[bert_config.vocab_size],
        #     initializer=tf.zeros_initializer()
        # )
        output_bias = tf.Variable(
            initial_value=tf.zeros([bert_config.vocab_size]),
            name='output_bias',
            shape=[bert_config.vocab_size],
            # initializer=tf.zeros_initializer()
        )
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids,
            depth=bert_config.vocab_size,
            dtype=tf.float32
        )

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    '''Get loss and log probs for the next sentence prediction'''

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    # with tf.variable_scope('cls/seq_relationship'):
    with tf.name_scope('cls/seq_relationship'):
        # output_weights = tf.get_variable(
        output_weights = tf.Variable(
            initial_value=modeling.generate_initial_value(
                shape=[2, bert_config.hidden_size],
                stddev=bert_config.initializer_range
            ),
            name='output_weights',
            shape=[2, bert_config.hidden_size],
            # initializer=modeling.create_initializer(bert_config.initializer_range)
        )
        # output_bias = tf.get_variable(
        output_bias = tf.Variable(
            initial_value=tf.zeros([2]),
            name='output_bias',
            shape=[2],
            # initializer=tf.zeros_initializer()
        )

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    '''Gathers the vectors at the specific positions over a minibatch'''
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files, max_seq_length, max_predictions_per_seq, is_training, num_cpu_threads=4):
    '''Creates an `input_fn` closure to be passed to TPUEstimator'''

    def input_fn(params):
        '''The actual input function'''
        batch_size = params['batch_size']

        name_to_features = {
            'input_ids':
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'input_mask':
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'segment_ids':
                tf.io.FixedLenFeature([max_seq_length], tf.int64),
            'masked_lm_positions':
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'masked_lm_ids':
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'masked_lm_weights':
                tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            'next_sentence_labels':
                tf.io.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.list_files(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            d = d.interleave(
                map_func=tf.data.TFRecordDataset,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

            # the outputs need to be produced in deterministic order (sloppy = True)
            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            options = tf.data.Options()
            options.experimental_deterministic = True
            d = d.with_options(options)
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        # map and batch
        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        d = d.batch(
            batch_size=batch_size,
            drop_remainder=True if is_training else False
        )

        return d

    return input_fn


def _decode_record(record, name_to_features):
    '''Decodes a record to a TensorFlow example'''
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example
