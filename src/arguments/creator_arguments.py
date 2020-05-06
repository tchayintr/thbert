from pathlib import Path

from arguments.arguments import ArgumentLoader



class CreatorArgumentLoader(ArgumentLoader):
    def parse_args(self):
        return super().parse_args()


    def get_full_parser(self):
        parser = super().get_full_parser()

        ### model parameters
        ### options for model architecture and parameters
        # common
        parser.add_argument('--iterations_per_loop', type=int, default=1000, help='The number steps to make in each estimator call (Default: 1000)')
        parser.add_argument('--max_eval_steps', type=int, default=100, help='Maximum number of eval steps (Default: 100)')
        parser.add_argument('--num_train_steps', type=int, default=100000, help='The number of training steps (Default: 100000)')
        parser.add_argument('--num_warmup_steps', type=int, default=10000, help='The number of warmup steps (Default: 10000)')
        parser.add_argument('--use_one_hot_embeddings', action='store_true', help='Use one-hot word embeddings or tf.embedding_lookup() for the word embeddings (On the TPU, it is much faster if this is True, on the CPU or GPU, it is faster if this is False)')

        # BERT configurations
        parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1, help='Dropout ratio for attentions (Default: 0.1)')
        parser.add_argument('--directionality', type=str.lower, default='bidi', help='Directionality of networks (Default: bidi)')
        parser.add_argument('--hidden_act', type=str.lower, default='gelu', help='Activation function for hidden layers (Default: gelu)')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout ratio for hidden layers (Default: 0.1)')
        parser.add_argument('--hidden_size', type=int, default=768, help='Dimension (size) of hidden layers (Default: 768)')
        parser.add_argument('--initializer_range', type=float, default=0.02, help='(Default: 0.02)')
        parser.add_argument('--intermediate_size', type=int, default=3072, help='(Default: 3072)')
        parser.add_argument('--max_position_embeddings', type=int, default=512, help='Maximum dimension (size) of embedding layers (Default: 512)')
        parser.add_argument('--num_attention_heads', type=int, default=12, help='The number of attention heads (Default: 12)')
        parser.add_argument('--num_hidden_layers', type=int, default=12, help='The number of hidden layers (Default: 12)')
        parser.add_argument('--pooler_fc_size', type=int, default=768, help='(Default: 768)')
        parser.add_argument('--pooler_num_attention_heads', type=int, default=12, help='(Default: 12)')
        parser.add_argument('--pooler_num_fc_layers', type=int, default=3, help='(Default: 3)')
        parser.add_argument('--pooler_size_per_head', type=int, default=128, help='(Default: 128)')
        parser.add_argument('--pooler_type', type=str.lower, default='first_token_transform', help='(Default: first_token_transform)')
        parser.add_argument('--type_vocab_size', type=int, default=2, help='The vocabulary size of the `token_type_ids` passed into BertModel (Default: 2)')
        parser.add_argument('--vocab_size', type=int, default=32000, help='Size of vocabulary (Default: 32000)')
        parser.add_argument('--bert_config_path', type=Path, default=None, help='File path of BERT configuration (High priority parameters)')

        # tpu options (depatched)
        parser.add_argument('--use_tpu', action='store_true', help='[DISPATCHED] Use TPU')
        parser.add_argument('--tpu_name', type=str.lower, default=None, help='[DISPATCHED] The Cloud TPU to use for training (name or address) (Default: None)')
        parser.add_argument('--tpu_zone', type=str.lower, default=None, help='[DISPATCHED] GCE zone where the Cloud TPU is located (Default: None)')
        parser.add_argument('--gcp_project', type=str.lower, default=None, help='[DISPATCHED] Project name for the Cloud TPU-enabled project (Default: None)')
        parser.add_argument('--master', type=str.lower, default=None, help='[DISPATCHED] TensorFlow master URL (Default: None)')
        parser.add_argument('--num_tpu_cores', type=int, default=8, help='[DISPATCHED] Total number of TPU cores to use (Default: 8)')

        return parser


    def get_minimum_parser(self, args):
        parser = super().get_minimum_parser(args)

        # options for model architecture and parameters
        self.add_creator_basic_options(parser, args)
        self.add_bert_configs_options(parser, args)

        # depatched options
        self.add_depatched_options(parser, args)

        return parser


    def add_creator_basic_options(self, parser, args):
        parser.add_argument('--iterations_per_loop', type=int, default=args.iterations_per_loop)
        parser.add_argument('--max_eval_steps', type=int, default=args.max_eval_steps)
        parser.add_argument('--num_train_steps', type=int, default=args.num_train_steps)
        parser.add_argument('--num_warmup_steps', type=int, default=args.num_warmup_steps)
        parser.add_argument('--use_one_hot_embeddings', action='store_true', default=args.use_one_hot_embeddings)


    def add_bert_configs_options(self, parser, args):
        parser.add_argument('--attention_probs_dropout_prob', type=float, default=args.attention_probs_dropout_prob)
        parser.add_argument('--directionality', default=args.directionality)
        parser.add_argument('--hidden_act', default=args.hidden_act)
        parser.add_argument('--hidden_dropout_prob', type=float, default=args.hidden_dropout_prob)
        parser.add_argument('--hidden_size', type=int, default=args.hidden_size)
        parser.add_argument('--initializer_range', type=float, default=args.initializer_range)
        parser.add_argument('--intermediate_size', type=int, default=args.intermediate_size)
        parser.add_argument('--max_position_embeddings', type=int, default=args.max_position_embeddings)
        parser.add_argument('--num_attention_heads', type=int, default=args.num_attention_heads)
        parser.add_argument('--num_hidden_layers', type=int, default=args.num_hidden_layers)
        parser.add_argument('--pooler_fc_size', type=int, default=args.pooler_fc_size)
        parser.add_argument('--pooler_num_attention_heads', type=int, default=args.pooler_num_attention_heads)
        parser.add_argument('--pooler_num_fc_layers', type=int, default=args.pooler_num_fc_layers)
        parser.add_argument('--pooler_size_per_head', type=int, default=args.pooler_size_per_head)
        parser.add_argument('--pooler_type', default=args.pooler_type)
        parser.add_argument('--type_vocab_size', type=int, default=args.type_vocab_size)
        parser.add_argument('--vocab_size', type=int, default=args.vocab_size)
        parser.add_argument('--bert_config_path', type=Path, default=args.bert_config_path)


    def add_depatched_options(self, parser, args):
        parser.add_argument('--use_tpu', action='store_true', default=args.use_tpu)
        parser.add_argument('--tpu_name', default=args.tpu_name)
        parser.add_argument('--tpu_zone', default=args.tpu_zone)
        parser.add_argument('--gcp_project', default=args.gcp_project)
        parser.add_argument('--master', default=args.master)
        parser.add_argument('--num_tpu_cores', type=int, default=args.num_tpu_cores)
