from core.rl_paradigm import BaseParadigm
import flashbax as fbx
from agents.base_trainer import BaseTrainer
from typing import Tuple, Any
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
import wandb
import jax
import omegaconf



class OfflineRLParadigm(BaseParadigm):
    def __init__(
            self,
            args: Any,
            trainer: BaseTrainer, 
            evaluation_env: Tuple, # d4rl env collection
            replay_buffer: Tuple[TrajectoryBuffer, Any], # inited replay buffer and its state
            batch_size: int = 256,
            start_epoch: int = -1000, # negative epochs are offline, positive epochs are online
            end_epoch: int = 0,
            total_steps: int = 1000000,
            num_eval_episodes_per_epoch: int = 5,
            num_training_steps_per_epoch: int = 1000,
        ):
        super().__init__(
            args, 
            trainer, 
            batch_size,
            start_epoch,
            end_epoch,
            total_steps
        )

        self.eval_env_collection = evaluation_env
        assert len(self.eval_env_collection) == self.args.num_seeds, 'number of envs not equals to that of multiple runs'
        self.buffer, self.buffer_state  = replay_buffer[0], replay_buffer[1]

        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch 
        self.online_finetune = (self.end_epoch > 0)
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.num_training_steps_per_epoch = num_training_steps_per_epoch 

        assert self.start_epoch < 0, 'start_epoch should be negative for offline RL'
        assert (self.end_epoch - self.start_epoch) * self.num_training_steps_per_epoch == self.total_steps, 'total training steps mismatch '

    def get_init_states(self, rng):
        # init offline rl trainer
        trainer_state = self.trainer.init_trainer_state(rng)
        # duplicate buffer state from offline datasets 
        buffer_state = jax.tree_util.tree_map()
        return trainer_state, buffer_state

    def make_train(self, exp_dir, wandb_mode=False):
        WANDB_MODE = wandb_mode
        # set wandb logger
        if WANDB_MODE: 
            self._wandb = wandb.init(
                project=self.args.project,
                tags=[
                    'offline',
                    'd4rl',
                    f"jax_{jax.__version__}",
                ],
                name=f"{self.args.env.name}_{self.args.trainer.exp_prefix}",
                dir=exp_dir,
                config=omegaconf.OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True),
                # mode='online'
            )

        def train(rng, pseudo_seed):

            

            pass

        

        return train


    def epoch_train(self, ):
        """ train loop per epoch: """
        # TODO: 'offline training'
        pass

    def policy_evaluate(self, trainer_state, idx, rng):
        pass
