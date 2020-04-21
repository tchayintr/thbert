import sys

from data_loaders import creation_data_loader
from pretrainers import creation_pretrainer
from trainers.trainer import Trainer



class CreatorTrainerBase(Trainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def show_data_info(self, data_type):
        if data_type == 'valid':
            data = self.valid
        elif data_type == 'dic':
            data = self.dic
        else:
            data = self.train

        self.log('### {} info'.format(data_type))
        self.log('# length: {}'.format(len(data)))
        self.log()


    def show_training_data(self):
        train = self.train
        valid = self.valid
        vocabs = self.vocabs

        train_size = len(train) if train else 0
        valid_size = len(valid) if valid else 0
        vocabs_size = len(vocabs) if vocabs else 0

        self.log('### Loaded data')
        self.log('# train: {} ...'.format(train_size))
        self.log('# valid: {} ...'.format(valid_size))
        self.log('# vocabulary: {} ...'.format(vocabs_size))
        self.log()

        self.report('[INFO] data length: train={} valid={}, vocab={}'.format(train_size, valid_size, vocabs_size))



class CreatorTrainer(CreatorTrainerBase):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)


    def init_model(self):
        super().init_model()


    def load_model(self):
        super().load_model()


    def init_hyperparameters(self):
        self.hparams = {
            'attention_probs_dropout_prob': self.args.attention_probs_dropout_prob,
            'batch_size': self.args.batch_size,
            'bert_config_path': self.args.bert_config_path,
            'checkpoint_path': self.args.checkpoint_path,
            'checkpoints_steps': self.args.checkpoints_steps,
            'dic_data': self.args.dic_data,
            'directionality': self.args.directionality,
            'dupe_factor': self.args.dupe_factor,
            'eval_result_path': self.args.eval_result_path,
            'execute_mode': self.args.execute_mode,
            'hidden_act': self.args.hidden_act,
            'hidden_dropout_prob': self.args.hidden_dropout_prob,
            'hidden_size': self.args.hidden_size,
            'initializer_range': self.args.initializer_range,
            'intermediate_size': self.args.intermediate_size,
            'learning_rate': self.args.learning_rate,
            'lowercasing': self.args.lowercasing,
            'mask_whole_word': self.args.mask_whole_word,
            'masked_lm_prob': self.args.masked_lm_prob,
            'max_eval_steps': self.args.max_eval_steps,
            'max_position_embeddings': self.args.max_position_embeddings,
            'max_predictions': self.args.max_predictions,
            'max_sequence_length': self.args.max_sequence_length,
            'normalize_digits': self.args.normalize_digits,
            'num_attention_heads': self.args.num_attention_heads,
            'num_hidden_layers': self.args.num_hidden_layers,
            'num_train_steps': self.args.num_train_steps,
            'num_warmup_steps': self.args.num_warmup_steps,
            'output_model_path': self.args.output_model_path,
            'pooler_fc_size': self.args.pooler_fc_size,
            'pooler_num_attention_heads': self.args.pooler_num_attention_heads,
            'pooler_num_fc_layers': self.args.pooler_num_fc_layers,
            'pooler_size_per_head': self.args.pooler_size_per_head,
            'pooler_type': self.args.pooler_type,
            'random_seed': self.args.random_seed,
            'short_seq_prob': self.args.short_seq_prob,
            'start_time': self.start_time,
            'summary_steps': self.args.summary_steps,
            'tfrecord_path': self.args.tfrecord_path,
            'train_data': self.args.train_data,
            'type_vocab_size': self.args.type_vocab_size,
            'use_one_hot_embeddings': self.args.use_one_hot_embeddings,
            'use_tpu': self.args.use_tpu,
            'valid_data': self.args.valid_data,
            'vocab_path': self.args.vocab_path,
            'vocab_model': self.args.vocab_model,
            'vocab_size': self.args.vocab_size,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def update_hyperparameter(self, key, value):
        self.hparams[key] = value


    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'batch_size' or
                    key == 'checkpoints_steps' or
                    key == 'dupe_factor' or
                    key == 'hidden_size' or
                    key == 'intermediate_size' or
                    key == 'max_eval_steps' or
                    key == 'max_position_embeddinags' or
                    key == 'max_predictions' or
                    key == 'max_sequence_length' or
                    key == 'num_attention_heads' or
                    key == 'num_hidden_layers' or
                    key == 'num_train_steps' or
                    key == 'num_warmup_steps' or
                    key == 'pooler_fc_size' or
                    key == 'pooler_num_attention_heads' or
                    key == 'pooler_num_fc_layers' or
                    key == 'pooler_size_per_head' or
                    key == 'random_seed' or
                    key == 'summary_steps' or
                    key == 'train_batch_size' or
                    key == 'type_vocab_size' or
                    key == 'vocab_size'
                ):
                    val = int(val)

                elif (key == 'attention_probs_dropout_prob' or
                      key == 'hidden_dropout_prob' or
                      key == 'initializer_range' or
                      key == 'learning_rate' or
                      key == 'masked_lm_prob' or
                      key == 'short_seq_prob'
                ):
                    val = float(val)

                elif (key == 'lowercasing' or
                      key == 'mask_whole_word' or
                      key == 'normalize_digits' or
                      key == 'use_one_hot_embeddings' or
                      key == 'use_tpu'
                ):
                    val = (val.lower() == 'true')

                elif (key == 'directionality' or
                      key == 'hidden_act' or
                      key == 'pooler_type'
                ):
                    val = val.lower()

                hparams[key] = val

        self.hparams = hparams
        self.log('Load hyperparameters: {}'.format(hparams_path))
        self.show_hyperparameters()


    def setup_data_loader(self):
        self.data_loader = creation_data_loader.CreationDataLoader(
            lowercasing=self.hparams['lowercasing'],
            normalize_digits=self.hparams['normalize_digits'],
        )


    def setup_pretrainer(self):
        if self.args.execute_mode == 'train':
            input_train_file = self.hparams['train_data']
            input_valid_file = None
        elif self.args.execute_mode == 'eval':
            input_train_file = None
            input_valid_file = self.hparams['valid_data']
        elif self.args.execute_mode == 'hybrid':
            input_train_file = self.hparams['train_data']
            input_valid_file = self.hparams['valid_data']

        self.pretrainer = creation_pretrainer.CreationPretrainer(
            input_train_file=input_train_file,
            input_valid_file=input_valid_file,
            output_record_path=self.hparams['tfrecord_path'],
            vocab_data=self.hparams['vocab_data_path'],
            do_lower_case=self.hparams['lowercasing'],
            do_whole_word=self.hparams['mask_whole_word'],
            max_seq_length=self.hparams['max_sequence_length'],
            max_predictions_per_seq=self.hparams['max_predictions'],
            masked_lm_prob=self.hparams['masked_lm_prob'],
            random_seed=self.hparams['random_seed'],
            dupe_factor=self.hparams['dupe_factor'],
            short_seq_prob=self.hparams['short_seq_prob'],
            attention_probs_dropout_prob=self.hparams['attention_probs_dropout_prob'],
            bert_config_path=self.hparams['bert_config_path'],
            directionality=self.hparams['directionality'],
            hidden_act=self.hparams['hidden_act'],
            hidden_dropout_prob=self.hparams['hidden_dropout_prob'],
            hidden_size=self.hparams['hidden_size'],
            initializer_range=self.hparams['initializer_range'],
            intermediate_size=self.hparams['intermediate_size'],
            max_position_embeddings=self.hparams['max_position_embeddings'],
            num_attention_heads=self.hparams['num_attention_heads'],
            num_hidden_layers=self.hparams['num_hidden_layers'],
            pooler_fc_size=self.hparams['pooler_fc_size'],
            pooler_num_attention_heads=self.hparams['pooler_num_attention_heads'],
            pooler_num_fc_layers=self.hparams['pooler_num_fc_layers'],
            pooler_size_per_head=self.hparams['pooler_size_per_head'],
            pooler_type=self.hparams['pooler_type'],
            type_vocab_size=self.hparams['type_vocab_size'],
            vocab_size=self.hparams['vocab_size'],
            batch_size=self.hparams['batch_size'],
            checkpoint_path=self.hparams['checkpoint_path'],
            eval_result_path=self.hparams['eval_result_path'],
            learning_rate=self.hparams['learning_rate'],
            max_eval_steps=self.hparams['max_eval_steps'],
            model_path=self.hparams['output_model_path'],
            num_train_steps=self.hparams['num_train_steps'],
            num_warmup_steps=self.hparams['num_warmup_steps'],
            save_checkpoints_steps=self.hparams['checkpoints_steps'],
            save_summary_steps=self.hparams['summary_steps'],
            use_one_hot_embeddings=self.hparams['use_one_hot_embeddings'],
            use_tpu=self.hparams['use_tpu'],
            execute_mode=self.args.execute_mode,
            quiet=self.args.quiet,
            logger=self.logger,
            reporter=self.reporter,
        )
        self.pretrainer.gen_pretraining_data()
