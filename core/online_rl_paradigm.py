from core.rl_paradigm import BaseParadigm
from agents.base_trainer import BaseTrainer
import wandb
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from typing import Tuple, Any, Union
import jax
import jax.numpy as jnp
from utils.misc import Transition
from functools import partial
import omegaconf

class OnlineRLParadigm(BaseParadigm):
    def __init__(
            self, 
            args: Any,
            trainer: BaseTrainer, 
            exploration_env: Tuple[Any, Any],  # no init env and its params
            evaluation_env: Tuple[Any, Any], 
            expl_env_nums: int,
            eval_env_nums: int,
            replay_buffer: Tuple[TrajectoryBuffer, Any], # no init buffer, [buffer_class, init_fake_trans]
            batch_size: int, 
            start_epoch: int = 0,
            end_epoch: int = 1000,
            total_steps: int = 1000000,
            num_timesteps_per_epoch: int = 1000,
            vec_env_rollout_len: int = 1,
            num_trains_per_expl_step: Union[int, None] = None,
            num_eval_episodes_per_epoch: Union[int, None] = 5,
            min_num_steps_before_training=10000,
        ):
        super().__init__(
            args=args,
            trainer=trainer, 
            batch_size=batch_size,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            total_steps = total_steps
        )

        self.expl_env_nums = expl_env_nums
        self.eval_env_nums = eval_env_nums
        self.expl_env, self.expl_env_params = exploration_env[0], exploration_env[1]
        self.eval_env, self.eval_env_params = evaluation_env[0], evaluation_env[1]
        self.buffer, self.fake_trans  = replay_buffer[0], replay_buffer[1]

        self.batch_size = batch_size
        self.start_epoch = start_epoch 
        self.end_epoch = end_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.num_timesteps_per_epoch = num_timesteps_per_epoch
        self.vec_env_rollout_len = vec_env_rollout_len
        self.num_expl_steps_per_epoch = self.num_timesteps_per_epoch // (self.expl_env_nums * self.vec_env_rollout_len) # 'expl_env_nums' Envs explores and samples simultaneously
        if num_trains_per_expl_step is None:
            self.num_trains_per_expl_step = self.num_timesteps_per_epoch // self.num_expl_steps_per_epoch
        else:
            self.num_trains_per_expl_step = num_trains_per_expl_step
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.total_steps == (self.end_epoch - self.start_epoch) * self.num_timesteps_per_epoch, 'total timesteps by all epoch training not consistent with \'total_steps\''
        assert self.start_epoch >=0 and self.end_epoch > self.start_epoch, 'not satisify the epoch number setting in online RL'
        assert self.num_timesteps_per_epoch >= self.num_expl_steps_per_epoch, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def get_init_states(self, rng):
        # init expl_env, replay buffer and trainer
        rng_reset, rng_tr = jax.random.split(rng, 2)
        rng_reset = jax.random.split(rng_reset, self.expl_env_nums)
        obs, expl_env_state = self.expl_env.reset(rng_reset, self.expl_env_params) # obs: (expl_env_nums, obs_dim)
        # init replay buffer
        buffer_state = self.buffer.init(self.fake_trans)
        # init trainer
        trainer_state = self.trainer.init_trainer_state(rng_tr)
        return obs, expl_env_state, buffer_state, trainer_state

    def make_train(self, exp_dir, wandb_mode=True):
        WANDB_MODE = wandb_mode
        # set wandb logger
        if WANDB_MODE: 
            self._wandb = wandb.init(
                project=self.args.project,
                tags=[
                    'jaxa baselines',
                    # f'{self.args.env.backend} backend',
                    f"jax_{jax.__version__}",
                ],
                name=f"{self.args.env.name}_{self.args.trainer.exp_prefix}",
                dir=exp_dir,
                config=omegaconf.OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True),
                mode='online'
            )

        def train(rng, pseudo_seed):

            rng_init, rng_epochs, rng_last_eval = jax.random.split(rng, 3)
            init_obs, expl_env_state, buffer_state, trainer_state = self.get_init_states(rng_init)

            def _epoch_train_eval(runner_state, rng_epoch):
                obs, expl_env_state, buffer_state, trainer_state = runner_state
                # evaluation before epoch training
                rng_eval, rng_tr = jax.random.split(rng_epoch)
                mean_episode_returns = self.policy_evaluate(trainer_state, rng_eval, expl_env_state)
                # implement epoch training
                runner_state, total_infos = self.epoch_train(obs, expl_env_state, buffer_state, trainer_state, rng_tr)
                # logging infos
                epoch_idx = runner_state[-1].epoch_idx
                if WANDB_MODE:
                    def callback(total_infos, mean_episode_returns, pseudo_seed, epoch_idx):
                        total_infos = jax.tree_util.tree_map(lambda x: jnp.mean(x), total_infos)
                        total_infos['eval/mean_episode_returns'] = mean_episode_returns
                        self._wandb.log({f'seed_{pseudo_seed}/{k}': v for k, v in total_infos.items()}, step=epoch_idx*self.num_timesteps_per_epoch) 
                        # self.trainer.logger.info(f'Epoch-{epoch_idx}: Eval Returns:{mean_episode_returns}')
                    jax.debug.callback(callback, total_infos, mean_episode_returns, pseudo_seed, epoch_idx)

                jax.debug.print('Seed-{seed}-Epoch-{e_idx}: episode returns: {rets}', seed=pseudo_seed, e_idx=epoch_idx, rets=mean_episode_returns)
                return runner_state, None

            rng_epochs = jax.random.split(rng_epochs, self.end_epoch - self.start_epoch)
            runner_state, _ = jax.lax.scan(
                _epoch_train_eval,
                (init_obs, expl_env_state, buffer_state, trainer_state),
                rng_epochs
            )
            # last evaluation
            mean_episode_returns = self.policy_evaluate(runner_state[-1], rng_last_eval, runner_state[1])
            epoch_idx = runner_state[-1].epoch_idx
            if WANDB_MODE:
                def final_callback(mean_episode_returns, pseudo_seed, epoch_idx):
                    self._wandb.log({f'seed_{pseudo_seed}/eval/mean_episode_returns': mean_episode_returns}, step=(epoch_idx+1)*self.num_timesteps_per_epoch) 
                jax.debug.callback(final_callback, mean_episode_returns, pseudo_seed, epoch_idx)

            return {'trainer_state': runner_state[-1]}

        return train

    @partial(jax.jit, static_argnames=['self'])
    def policy_evaluate(self, trainer_state, rng, expl_env_state):
        rng_reset, rng_eval = jax.random.split(rng)
        rng_reset = jax.random.split(rng_reset, self.eval_env_nums)
        if self.args.env.obs_norm:
            statistic = (expl_env_state.mean, expl_env_state.var)
            statistic = jax.tree_util.tree_map(lambda x: jnp.repeat(x[0][None,], self.eval_env_nums, 0), statistic)
            obs, eval_env_state = self.eval_env.reset(rng_reset, self.eval_env_params, statistic)
        else:
            obs, eval_env_state = self.eval_env.reset(rng_reset, self.eval_env_params)
    
        total_returns = 0
        eval_episodes = 0
        def eval_step(eval_runner):
            eval_env_state, total_returns, eval_episodes, obs, rng = eval_runner
            rng, rng_a, rng_step = jax.random.split(rng, 3)
            actions = self.trainer.get_action(trainer_state, obs, rng_a, deterministic=True)
            rng_step = jax.random.split(rng_step, self.eval_env_nums)
            next_obs, eval_env_state, rewards, dones, info = self.eval_env.step(
                rng_step, eval_env_state, actions, self.eval_env_params
            )
            total_returns += jnp.dot(info['returned_episode_returns'], info['returned_episode'].astype(jnp.float32))
            eval_episodes += jnp.sum(info["returned_episode"].astype(jnp.int32))
            eval_runner = (eval_env_state, total_returns, eval_episodes, next_obs, rng)
            return eval_runner

        eval_runner = jax.lax.while_loop(
            cond_fun=lambda x: x[2] <= self.num_eval_episodes_per_epoch, 
            body_fun=eval_step, 
            init_val=(eval_env_state, total_returns, eval_episodes, obs, rng_eval),
        )

        return eval_runner[1] / eval_runner[2]

    @partial(jax.jit, static_argnames=['self'])
    def epoch_train(self, obs, expl_env_state, buffer_state, trainer_state, rng):
        """ Online Training for One Epoch """
        def _step_and_train(runner_state, rng_step_tr):
            """ Env step once & agent train once"""
            def _rollout(runner_state, rng_rout):
                rng_a, rng_step = jax.random.split(rng_rout)
                rng_step = jax.random.split(rng_step, self.expl_env_nums)
                obs, expl_env_state, buffer_state, trainer_state = runner_state
                actions = self.trainer.get_action(trainer_state, obs, rng_a, deterministic=False)
                next_obs, expl_env_state, rewards, dones, info = self.expl_env.step(
                    rng_step, expl_env_state, actions, self.expl_env_params
                    )
                trans = Transition(obs=obs, action=actions, reward=rewards, done=dones)
                buffer_state = self.buffer.add(buffer_state, trans)
                return (next_obs, expl_env_state, buffer_state, trainer_state), info 
            
            rng_step, rng_tr = jax.random.split(rng_step_tr)
            rng_routs = jax.random.split(rng_step, self.vec_env_rollout_len)
            (obs, expl_env_state, buffer_state, trainer_state), infos = jax.lax.scan(_rollout, runner_state, rng_routs)

            can_learn = (
                (self.buffer.can_sample(buffer_state))
                & (
                    jnp.sum(infos['timestep'][-1]) > self.min_num_steps_before_training
                )
            )

            trainer_state, total_infos = jax.lax.cond(
                can_learn,
                lambda buffer_state, trainer_state, rng: self._train_once(buffer_state, trainer_state, rng),
                lambda buffer_state, trainer_state, rng: (trainer_state, self.trainer.null_total_infos), # do nothing
                buffer_state,
                trainer_state, 
                rng_tr,
            )
            runner_state = (obs, expl_env_state, buffer_state, trainer_state)
            return runner_state, total_infos

        rng_epoch = jax.random.split(rng, self.num_expl_steps_per_epoch)
        runner_state = (obs, expl_env_state, buffer_state, trainer_state)
        (obs, expl_env_state, buffer_state, trainer_state), total_infos = jax.lax.scan(_step_and_train, runner_state, rng_epoch)
        trainer_state = trainer_state.replace(epoch_idx=trainer_state.epoch_idx+1)

        return (obs, expl_env_state, buffer_state, trainer_state), total_infos

    def _train_once(self, buffer_state, trainer_state, rng):
        batch_rng = jax.random.split(rng, (self.num_trains_per_expl_step, 2))
        for i in range(self.num_trains_per_expl_step):
            batch_data = self.buffer.sample(buffer_state, batch_rng[i][0]).experience
            trainer_state, total_infos = self.trainer.update(trainer_state, batch_data, batch_rng[i][1])
        return trainer_state, total_infos

            
        
