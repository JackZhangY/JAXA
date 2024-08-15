from core.rl_paradigm import BaseParadigm
from agents.base_trainer import BaseTrainer
import flashbax as fbx
import gtimer as gt
import swanlab
from typing import Tuple, Any
import chex, jax
import jax.numpy as jnp
from utils.misc import Transition
from functools import partial


class OnlineRLParadigm(BaseParadigm):
    def __init__(
            self, 
            trainer: BaseTrainer, 
            rng: chex.PRNGKey,
            exploration_env: Tuple[Any, Any], 
            evaluation_env: Tuple[Any, Any], 
            expl_env_nums: int,
            eval_env_nums: int,
            batch_size: int, 
            replay_buffer: Tuple[Any, Any],
            start_epoch: int = 0,
            num_epochs: int = 1000,
            num_trains_per_epoch: int = 1000,
            # epoch_interval_per_eval: int = 2,
            # num_eval_steps_per_epoch: int | None = None,
            num_eval_episodes_per_epoch: int | None = 10,
            min_num_steps_before_training=10000,
        ):
        super().__init__(
            trainer=trainer, 
            rng=rng, 
            exploration_env=exploration_env, 
            evaluation_env=evaluation_env, 
            expl_env_nums=expl_env_nums, 
            eval_env_nums=eval_env_nums, 
            start_epoch=start_epoch,
            replay_buffer=replay_buffer
        )

        # self.env_nums = env_nums
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_eval_episodes_per_epoch = num_eval_episodes_per_epoch
        self.num_trains_per_epoch = num_trains_per_epoch
        # self.epoch_interval_per_eval = epoch_interval_per_eval
        self.num_expl_steps_per_epoch = self.num_trains_per_epoch // self.expl_env_nums # 'expl_env_nums' Envs explores and samples simultaneously
        self.num_trains_per_expl_step = self.num_trains_per_epoch // self.num_expl_steps_per_epoch 
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_epoch >= self.num_expl_steps_per_epoch, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    @partial(jax.jit, static_argnames=['self'])
    def policy_evaluate(self, actor_state, rng):
        rng_reset, rng_eval = jax.random.split(rng)
        rng_reset = jax.random.split(rng_reset, self.eval_env_nums)
        obs, eval_env_state = self.eval_env.reset(rng_reset, self.eval_env_params)

        total_returns = 0
        eval_episodes = 0
        def eval_step(eval_runner):
            eval_env_state, total_returns, eval_episodes, obs, rng = eval_runner
            rng, rng_a, rng_step = jax.random.split(rng, 3)
            actions = self.trainer.get_action(actor_state, obs, rng_a, deterministic=True)
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

    def train(self,):
        # init expl_env:
        rng, rng_reset = jax.random.split(self.rng)
        rng_reset = jax.random.split(rng_reset, self.expl_env_nums)
        obs, expl_env_state = self.expl_env.reset(rng_reset, self.expl_env_params)

        gt.reset()
        times_dict = None

        agent_state = self.trainer.agent_state
        buffer_state = self.buffer_state

        for epoch in gt.timed_for(range(self.start_epoch, self.num_epochs), save_itrs=True):
            rng, rng_eval = jax.random.split(rng)
            mean_episode_returns = self.policy_evaluate(actor_state=agent_state[1], rng=rng_eval)
            gt.stamp('Policy Evaluation')
            runner_state, total_infos = self.on_train(
                obs, expl_env_state, buffer_state, agent_state, rng
            )
            obs, expl_env_state, buffer_state, agent_state, rng = runner_state
            
            # TODO: swanlab记录变量，
            # swanlab.log({'eval/return': mean_episode_returns}, step=epoch * self.num_trains_per_epoch)
            # self.infos_logging(total_infos, step=(epoch+1) * self.num_trains_per_epoch)
            
            gt.stamp('One Epoch')
            times_dict = self._get_epoch_timings()
            self.trainer.logger.info(
                'Epoch-{}: Eval Policy: {:.2f} second(s), Policy Train: {:.2f}, and Eval Returns: {:.2f}'.format(
                    epoch, times_dict['time/Policy Evaluation (s)'], times_dict['time/One Epoch (s)'], mean_episode_returns
                )
            )
        # last evaluation after all training steps
        mean_episode_returns = self.policy_evaluate(agent_state[1], rng_eval)
        # swanlab.log({'eval/return': mean_episode_returns}, step=(epoch +1 ) * self.num_trains_per_epoch)
        gt.stamp('Last Policy Evaluation')
        self.trainer.logger.info('Eval Returns at last epoch: {}'.format(mean_episode_returns))
        self.trainer.logger.info(
            'Total time cost: {}'.format(gt.get_times().total)
        )
        #  after training, update agent state
        self.trainer.update_agent_state(agent_state)


    @partial(jax.jit, static_argnames=['self'])
    def on_train(self, obs, expl_env_state, buffer_state, agent_state, rng):
        """ Online Training for One Epoch """
        def _step_and_train(runner_state, null):
            """ Env step once & agent train once"""
            obs, expl_env_state, buffer_state, agent_state, rng = runner_state
            rng, rng_a, rng_step = jax.random.split(rng, 3)
            actions = self.trainer.get_action(agent_state[1], obs, rng_a, deterministic=False)
            rng_step = jax.random.split(rng_step, self.expl_env_nums)
            next_obs, expl_env_state, rewards, dones, info = self.expl_env.step(
                rng_step, expl_env_state, actions, self.expl_env_params
                )
            trans = Transition(obs=obs, action=actions, reward=rewards, done=dones)
            buffer_state = self.buffer.add(buffer_state, trans)

            can_learn = (
                (self.buffer.can_sample(buffer_state))
                & (
                    jnp.sum(info['timestep']) > self.min_num_steps_before_training
                )
            )

            rng, rng_ = jax.random.split(rng) 
            agent_state, total_infos = jax.lax.cond(
                can_learn,
                lambda buffer_state, agent_state, rng: self._train_once(buffer_state, agent_state, rng),
                lambda buffer_state, agent_state, rng: (agent_state, self.trainer.null_total_infos), # do nothing
                buffer_state,
                agent_state, 
                rng_,
            )

            runner_state = (next_obs, expl_env_state, buffer_state, agent_state, rng)
            return runner_state, total_infos
        
        runner_state = (obs, expl_env_state, buffer_state, agent_state, rng)
        runner_state, total_infos = jax.lax.scan(
            _step_and_train, runner_state, None, self.num_expl_steps_per_epoch 
        )
        return runner_state, total_infos

    def _train_once(self, buffer_state, agent_state, rng):
        for _ in range(self.num_trains_per_expl_step):
            rng_sample, rng_update = jax.random.split(rng)
            batch_data = self.buffer.sample(buffer_state, rng_sample).experience
            agent_state, total_infos = self.trainer.update(agent_state, batch_data, rng_update)
        return agent_state, total_infos

    def infos_logging(self, total_infos, step):
        for k, v in total_infos.items():
            swanlab.log({k: jnp.mean(v)}, step=step)
            
        
