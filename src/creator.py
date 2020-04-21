from arguments import creator_arguments
from core import Core
from trainers import creator_trainer
import util



class Creator(Core):
    def __init__(self):
        super().__init__()


    def get_timer(self):
        timer = util.Time()
        return timer


    def get_args(self):
        parser = creator_arguments.CreatorArgumentLoader()
        args = parser.parse_args()
        return args


    def get_trainer(self, args):
        trainer = creator_trainer.CreatorTrainer(args)
        return trainer



if __name__ == '__main__':
    creator = Creator()
    timer = creator.get_timer()
    timer.start()
    creator.run()
    timer.stop()
    print('### Elapsed time: {} seconds'.format(timer.elapsed))
