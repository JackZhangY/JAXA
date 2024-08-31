from functools import partial
from agents.base_trainer import BaseTrainer
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import os
import jax, chex, optax
import jax.numpy as jnp
import distrax
from utils.misc import TrainState, TargetTrainState, SACTrainerState, CriticTrainerState
from typing import Dict
from utils.networks import *

class NAFTrainer(BaseTrainer):

    def __init__(
            self, 
            name: str,
            log_dir: str, 
            ens_num: int, 
            vf_kwargs: FrozenDict, 
            mu_kwargs: FrozenDict,
            L_kwargs: FrozenDict,
            dummy_obs: chex.Array, 
            dummy_action: chex.Array,
            opt_kwargs: Dict,
            action_noise: float,
            reward_scaling: float = 1.0,
            tau: float = 0.01,
            discount: float = 0.99,
            exp_prefix: str = ''
        ) -> None:
        super().__init__(name, log_dir)

        self.dummy_obs = dummy_obs
        self.dummy_action = dummy_action 
        self.obs_dim = dummy_obs.shape[-1]# FlattenObservationWrapper in default
        self.action_dim = dummy_action.shape[-1]
        self.action_noise = action_noise
        self.reward_scaling = reward_scaling
        self.tau = tau
        self.discount = discount 
        self.opt_kwargs = opt_kwargs
        # build trainer network
        self.quad_critic_net = jax.vmap(
            NAFQuadraticAC,
            in_axes = None, out_axes = 0,
            variable_axes = {'params': 0},
            split_rngs = {'params': True},
            axis_size = ens_num,
            method = ['__call__', 'get_mu', 'get_v']
        )(
            action_dim = self.action_dim,
            V_hidden_cfg = vf_kwargs['hidden_cfg'],
            L_hidden_cfg = L_kwargs['hidden_cfg'],
            mu_hidden_cfg = mu_kwargs['hidden_cfg']
        )

    def init_trainer_state(self, rng):
        # init AC model state
        quad_critic_params = self.quad_critic_net.init(rng, self.dummy_obs, self.dummy_action) # 'in_axes' is None, so no need to infer 'axis_size' by input size  
        quad_critic_state = TargetTrainState.create(
            apply_fn=self.quad_critic_net.apply,
            params=quad_critic_params,
            target_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), quad_critic_params),
            tx=self.set_optimizer(**self.opt_kwargs),
            n_updates=0,
        )
        return CriticTrainerState(critic_state=quad_critic_state, epoch_idx=0)


    def get_action(self, trainer_state, obs, rng, deterministic=False):
        if not deterministic:
            action = self.random_action(trainer_state.critic_state, obs, rng)
        else: 
            action = self.optimal_action(trainer_state.critic_state, obs, rng)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def random_action(self, quad_critic_state, obs, rng):
        rng_a, rng_noise = jax.random.split(rng, 2)
        action = quad_critic_state.apply_fn(quad_critic_state.params, obs, method='get_mu') #(ens_num, bs, act_dim)
        action = jax.random.choice(key=rng_a, a=action, axis=0)
        action += self.action_noise * jax.random.normal(rng_noise, shape=action.shape, dtype=action.dtype)
        return action 

    @partial(jax.jit, static_argnames=['self'])
    def optimal_action(self, quad_critic_state, obs, rng):
        action = quad_critic_state.apply_fn(quad_critic_state.params, obs, method='get_mu')
        action = jax.random.choice(key=rng, a=action, axis=0)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def update(self, trainer_state, batch_data, rng):
        """ implement an update of NAF model once """
        """
        Args:
            batch_data: flashbax.sample.experience
        """
        quad_critic_state, quad_critic_loss = self.update_quad_critic(
            trainer_state.critic_state, batch_data
        )        
        total_infos = {
            'tr/q_loss': quad_critic_loss, 
        }

        return CriticTrainerState(critic_state=quad_critic_state, epoch_idx=trainer_state.epoch_idx), total_infos

    def update_quad_critic(self, quad_critic_state, batch_data):
        next_obs = batch_data.second.obs # (bs, dim)
        rewards = batch_data.first.reward # (bs, )
        v_next = quad_critic_state.apply_fn(quad_critic_state.target_params, next_obs, method='get_v')
        v_next = jnp.min(v_next, axis=0) #(bs, 1)
        q_target = jax.lax.stop_gradient(rewards * self.reward_scaling + self.discount * (1. - batch_data.first.done) * v_next.squeeze())  #(bs, )

        def q_critic_loss(params):
            qs = quad_critic_state.apply_fn(params, batch_data.first.obs, batch_data.first.action) #(ens_num, bs, 1)
            loss = ((qs.squeeze(axis=-1) - q_target)**2).mean(axis=-1).sum()
            return loss
        
        quad_critic_loss, quad_critic_grad = jax.value_and_grad(q_critic_loss)(quad_critic_state.params)
        quad_critic_state = quad_critic_state.apply_gradients(grads=quad_critic_grad)
        quad_critic_state = quad_critic_state.replace(
            target_params=optax.incremental_update(
                quad_critic_state.params, quad_critic_state.target_params, self.tau
            ),
            n_updates=quad_critic_state.n_updates + 1
        )
        return quad_critic_state, quad_critic_loss
    
    
    @property
    def null_total_infos(self):
        """ should match with the format of 'total_info' in func 'update()' """
        infos = {'tr/q_loss': jnp.array(0.0)}
        return infos

    def save_trainer_state(self, trainer_state_outs, outs_size):
        all_critic_params = trainer_state_outs.critic_state.params
        for i in range(outs_size):
            model_params_dict = {
                'critic_params': jax.tree_util.tree_map(lambda x: x[i], all_critic_params)
            }
            save_dir = os.path.join(self.log_dir, f'final_model_vmap_{i}')
            self.save_model(model_params_dict, save_dir)