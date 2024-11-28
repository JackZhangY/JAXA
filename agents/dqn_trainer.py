
from functools import partial
from agents.base_trainer import BaseTrainer
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import os
import jax, chex, optax
import jax.numpy as jnp
from utils.misc import TargetTrainState, CriticTrainerState 
from typing import Dict, Union
from utils.networks import *
from utils.exploration import *

class DQNTrainer(BaseTrainer):

    def __init__(
            self, 
            name: str,
            log_dir: str, 
            qf_kwargs: FrozenDict, 
            dummy_obs: chex.Array, 
            action_dim: chex.Array,
            critic_opt_kwargs: Dict,
            exploration_kwargs: FrozenDict,
            reward_scaling: float = 1.0,
            reward_bias: float = 0.0,
            tau: float = 0.005,
            target_update_freq: int = 1000,
            discount: float = 0.99,
            exp_prefix: str = ''
        ) -> None:
        super().__init__(name, log_dir)

        self.dummy_obs = dummy_obs[None,]
        self.action_dim = action_dim
        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.discount = discount 
        self.critic_opt_kwargs = critic_opt_kwargs
        # build trainer network
        if len(self.dummy_obs.shape) > 2: # no 'FlattenObservation' input 
            self.q_critic = ConvQNet(action_dim=self.action_dim, **qf_kwargs)
        else:
            self.q_critic = MLPQNet(action_dim=self.action_dim, **qf_kwargs)
        self.exploration_epsilon = select_epsilon_exploration(exploration_kwargs['name'], exploration_kwargs['params'])

    def init_trainer_state(self, rng):
        # init Q network model state
        Q_critic_params = self.q_critic.init(rng, self.dummy_obs) 
        Q_critic_state = TargetTrainState.create(
            apply_fn=self.q_critic.apply,
            params=Q_critic_params,
            target_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), Q_critic_params),
            tx=self.set_optimizer(**self.critic_opt_kwargs),
            n_updates=0,
        )

        return CriticTrainerState(critic_state=Q_critic_state, epoch_idx=0)

    def get_action(self, trainer_state, obs, rng, deterministic=False):
        """
        Return:
            action: jnp.Array with shape: (n,)
        """
        if not deterministic:
            rng_eps, rng_act = jax.random.split(rng)
            epsilon  = jax.random.uniform(rng_eps)
            cur_epsilon = self.exploration_epsilon.eps_threshold(trainer_state.critic_state.n_updates)

            action = jax.lax.cond(
                epsilon < cur_epsilon,
                lambda trainer_state, obs, rng: self.random_action(obs, rng),
                lambda trainer_state, obs, rng: self.optimal_action(trainer_state.critic_state, obs),
                trainer_state, 
                obs,
                rng_act,
            )
        else: 
            action = self.optimal_action(trainer_state.critic_state, obs)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def random_action(self, obs, rng):
        """
        Args:
            obs: used to determine the sampleing shape
        """
        action = jax.random.randint(rng, shape=(obs.shape[0],), minval=0, maxval=self.action_dim)
        return action 

    @partial(jax.jit, static_argnames=['self'])
    def optimal_action(self, Q_critic_state, obs):
        q_value = Q_critic_state.apply_fn(Q_critic_state.params, obs)
        action = jnp.argmax(q_value, axis=-1) # (bs,)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def update(self, trainer_state, batch_data, ph_rng):
        """ implement an update of Q network once """
        """
        Args:
            batch_data: flashbax.sample.experience
        """

        critic_state, critic_infos = self.update_critic(trainer_state.critic_state, batch_data)        
        total_infos = {
            'tr/q_loss': critic_infos['q_loss'], 
            'tr/action_gap': critic_infos['action_gap'], 
        }
        trainer_state = trainer_state.replace(critic_state=critic_state)
        return trainer_state, total_infos


    def update_critic(self, Q_critic_state, batch_data):
        next_obs = batch_data.second.obs # (bs, *obs_dim)
        rewards = batch_data.first.reward # (bs, )
        target_q_next = Q_critic_state.apply_fn(Q_critic_state.target_params, next_obs)
        target_q_next = jnp.max(target_q_next, axis=-1) # (bs,)

        q_target = jax.lax.stop_gradient(
            self.reward_scaling * rewards + self.reward_bias + self.discount * (1. - batch_data.first.done) \
            * (target_q_next)
        )

        def q_loss_fun(params):
            qs = Q_critic_state.apply_fn(params, batch_data.first.obs) # (bs, act_dim)
            vs = jnp.max(qs, axis=-1, keepdims=True)
            mean_action_gap = jnp.mean(vs - qs)
            qs = jnp.take_along_axis(
                qs,
                batch_data.first.action[:, None], # (bs, 1)
                axis=-1
            )  
            loss = jnp.mean(optax.losses.huber_loss(qs.squeeze(), q_target)) # equivalent with SmoothL1Loss
            return loss, mean_action_gap
        (q_critic_loss, action_gap), q_critic_grad = jax.value_and_grad(q_loss_fun, has_aux=True)(Q_critic_state.params)
        Q_critic_state = Q_critic_state.apply_gradients(grads=q_critic_grad)
        Q_critic_state = Q_critic_state.replace(
            n_updates=Q_critic_state.n_updates + 1
        )
        Q_critic_state = jax.lax.cond(
            Q_critic_state.n_updates % self.target_update_freq == 0,
            lambda Q_critic_state: Q_critic_state.replace(
                target_params=optax.incremental_update(
                    Q_critic_state.params, Q_critic_state.target_params, self.tau
                )
            ),
            lambda Q_critic_state: Q_critic_state,
            Q_critic_state,
        )
        return Q_critic_state, {'q_loss': q_critic_loss, 'action_gap': action_gap}
        
    
    @property
    def null_total_infos(self):
        """ should match with the format of 'total_infos' in func 'update()' """
        infos = {
            'tr/q_loss': jnp.array(0.0),
            'tr/action_gap': jnp.array(0.0), 
        }
        return infos

    def save_trainer_state(self, trainer_state_outs, outs_size):
        all_critic_params = trainer_state_outs.critic_state.params
        
        for i in range(outs_size):
            model_params_dict = {
                'critic_params': jax.tree_util.tree_map(lambda x: x[i], all_critic_params)
            }
            save_dir = os.path.join(self.log_dir, f'final_model_vmap_{i}')
            self.save_model(model_params_dict, save_dir)