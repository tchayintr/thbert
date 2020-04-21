import os

import constants



class Core(object):
    def get_args(self):
        # to be implemented in sub-class
        return None


    def get_trainer(self, args):
        # to be implemented in sub-class
        return None


    def run(self):
        ################################
        # Make necessary directories

        if not os.path.exists(constants.LOG_DIR):
            os.mkdir(constants.LOG_DIR)
        if not os.path.exists(constants.HPARAM_DIR):
            os.makedirs(constants.HPARAM_DIR)
        if not os.path.exists(constants.VOCAB_DIR):
            os.makedirs(constants.VOCAB_DIR)
        if not os.path.exists(constants.PRETRAINING_DIR):
            os.makedirs(constants.PRETRAINING_DIR)
        if not os.path.exists(constants.CHECKPOINT_DIR):
            os.makedirs(constants.CHECKPOINT_DIR)
        if not os.path.exists(constants.EVAL_DIR):
            os.makedirs(constants.EVAL_DIR)

        ################################
        # Get arguments and initialize trainer

        args = self.get_args()
        trainer = self.get_trainer(args)

        ################################
        # Prepare gpu

        use_gpu = 'gpu' in args and args.gpu >= 0
        if use_gpu:
            os.environ[constants.CUDA_DEVICE] = str(args.gpu)

        ################################
        # Init/load hyperparameters

        if args.hparams_path:
            trainer.load_hyperparameters(args.hparams_path)
        else:
            trainer.init_hyperparameters()

        ################################
        # Load dataset and set up dic

        if args.execute_mode == 'train':
            trainer.load_training_data()
        elif args.execute_mode == 'eval':
            trainer.load_validation_data()
        elif args.execute_mode == 'hybrid':
            trainer.load_training_and_validation_data()

        ################################
        # Setup pretrainer

        trainer.setup_pretrainer()

        ################################
        # Run

        if args.execute_mode == 'train':
            trainer.run_train_mode()

        elif args.execute_mode == 'eval':
            trainer.run_eval_mode()

        elif args.execute_mode == 'hybrid':
            trainer.run_hybrid_mode()

        ################################
        # Terminate

        trainer.close()
