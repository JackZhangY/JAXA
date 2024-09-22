import jax.experimental
from core.rl_paradigm import BaseParadigm
from functools import partial
from agents.base_trainer import BaseTrainer
from typing import Tuple, Any
from replay_buffers.d4rl_buffer import D4RLBufferState, D4RLBuffer
import wandb
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf


class OfflineRLParadigm(BaseParadigm):
    def __init__(
            self,
            args: Any,
            trainer: BaseTrainer, 
            evaluation_env: Tuple, # d4rl env collection
            replay_buffer: Tuple[D4RLBuffer, D4RLBufferState], # inited replay buffer and its state
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

        self.eval_vec_env = evaluation_env
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
        trainer_state = self.trainer.init_trainer_state(rng)
        return trainer_state

    def make_train(self, exp_dir, wandb_mode=False):
        # set wandb logger
        if wandb_mode: 
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

        def train(rng):
            rng_init, rng_epochs = jax.random.split(rng)
            trainer_state = self.get_init_states(rng_init)

            # first evaluation before training
            self.policy_evaluate(trainer_state, self.trainer.null_total_infos)

            def _epoch_train_eval(trainer_state, rng_epoch):
                trainer_state, scanned_total_infos = self.epoch_train(trainer_state, rng_epoch)
                mean_total_infos = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), scanned_total_infos)
                self.policy_evaluate(trainer_state, mean_total_infos)
                return trainer_state, None
            
            rng_epochs = jax.random.split(rng_epochs, self.end_epoch - self.start_epoch)
            trainer_state, _ = jax.lax.scan(
                _epoch_train_eval,
                trainer_state,
                rng_epochs
            )

            return {'trainer_state': trainer_state}
        
        return train

    @partial(jax.jit, static_argnames=['self'])
    def epoch_train(self, trainer_state, rng_tr):
        """ train loop per epoch: """

        def _train_once(trainer_state, rng_tr):
            rng_sample, rng_update = jax.random.split(rng_tr)
            batch_data = self.buffer.sample(self.buffer_state, rng_sample)
            trainer_state, total_infos = self.trainer.update(batch_data, trainer_state, rng_update)

            return trainer_state, total_infos

        rng_tr = jax.random.split(rng_tr, self.num_training_steps_per_epoch)
        trainer_state, scanned_total_infos = jax.lax.scan(_train_once, trainer_state, rng_tr) #; Dict(array:(num_training_steps,))
        trainer_state = trainer_state.replace(epoch_idx = trainer_state.epoch_idx + 1.)
        return trainer_state, scanned_total_infos

    def policy_evaluate(self, trainer_state, total_infos):
        policy_fn = lambda actor_state, obs: actor_state.apply_fn(actor_state.params, obs)[0]
        epoch_idx = trainer_state.epoch_idx
        def eval_callback(actor_state, epoch_idx, total_infos):
            vec_eval_episodes = self.num_eval_episodes_per_epoch // self.args.env.num_eval_envs

            episode_returns = np.zeros(self.args.env.num_eval_envs, dtype=np.float32)
            episode_counts = np.zeros(self.args.env.num_eval_envs, dtype=np.int32)
            cum_episode_returns = np.zeros(self.args.env.num_eval_envs, dtype=np.float32)

            obs = self.eval_vec_env.reset()
            while (episode_counts < vec_eval_episodes).any():
                actions = policy_fn(actor_state, obs)
                obs, rewards, dones, infos = self.eval_vec_env.step(actions)
                episode_counts = episode_counts + dones.astype(np.int32)
                new_episode_returns = episode_returns + rewards
                cum_episode_returns += new_episode_returns * dones.astype(np.float32)
                episode_returns = new_episode_returns * (1. - dones) 

            mean_episode_return =  cum_episode_returns.sum() / episode_counts.sum()
            normalized_score = self.eval_vec_env.get_normalized_score(mean_episode_return) * 100

            self.trainer.logger.info(f'Epoch-{epoch_idx}: Normalized Scores:{normalized_score}')
            if self.args.wandb_mode:
                total_infos['eval/normalized_score'] = normalized_score
                self._wandb.log({f'{k}': v for k, v in total_infos.items()}, step=epoch_idx*self.num_training_steps_per_epoch) 
        return jax.debug.callback(eval_callback, trainer_state.actor_state, epoch_idx, total_infos)
        