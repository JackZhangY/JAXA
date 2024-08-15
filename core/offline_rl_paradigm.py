from core.rl_paradigm import BaseParadigm
import flashbax as fbx
from agents.base_trainer import BaseTrainer
import swanlab
import gtimer as gt



class OfflineRLParadigm(BaseParadigm):
    def __init__(
            self,
            trainer: BaseTrainer, 
            exploration_env, 
            evaluation_env, 
            replay_buffer: fbx.trajectory_buffer,
            total_training_steps: int,
            batch_size: int,
            max_path_length: int,
            num_epochs: int,
            num_eval_steps_per_epoch: int,
            num_expl_steps_per_train_loop: int,
            num_trains_per_train_loop: int,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            num_epochs_per_log_interval=1,
            online_finetune=False, # whether to implement offline2online
            save_best=False,
            log_dir=None
                 ):
        super().__init__(trainer, exploration_env, evaluation_env, replay_buffer)

        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.num_epochs_per_log_interval = num_epochs_per_log_interval
        self.online_finetune = online_finetune
        self.save_best = save_best
        self.log_dir = log_dir
        self.epoch = None
        self.cur_best = -float('inf')

        assert int((self.num_epochs - self._start_epoch) * self.num_train_loops_per_epoch * self.num_trains_per_train_loop)\
               == int(self.total_training_steps), 'mismatch of total training steps indicated in \'trainer\' and \'algorithm\''
        assert self._start_epoch < 0 and self.num_epochs >= 0, 'not satisfy epoch setting for offline(2online) RL'
        if self.online_finetune:
            assert self.num_epochs > 0, 'not satisfy epoch setting for offline2online RL'


    def train(self):
        # self.trainer.logger.info('')
        # swanlab.log()
        """epoch loop: Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.epoch < 0:
            # if self.epoch == self._start_epoch or (self.epoch + 1) % self.num_epochs_per_log_interval == 0:
            #     self._begin_epoch(self.epoch)
                self.off_train()
            else: 
                self.on_train()

            # if self.epoch == self._start_epoch or (self.epoch + 1)% self.num_epochs_per_log_interval == 0:
            #     self._end_epoch(self.epoch)

    def on_train(self):
        """ train loop per epoch: """
        # TODO: 'offline training'
        pass
        

    def off_train(self):
        # TODO: 'online finetune training'
        pass