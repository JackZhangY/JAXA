from functools import partial
from agents.base_trainer import BaseTrainer
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import os
import jax, chex, optax
import jax.numpy as jnp
import distrax
from utils.misc import TrainState, TargetTrainState, SACTrainerState
from typing import Dict
from utils.networks import *

class SACTrainer(BaseTrainer):

    def __init__(
            self, 
            name: str,
            log_dir: str, 
            qf_kwargs: FrozenDict, 
            actor_kwargs: FrozenDict,
            dummy_obs: chex.Array, 
            dummy_action: chex.Array,
            opt_kwargs: Dict,
            auto_finetune_temp: bool = True,
            reward_scaling: float = 1.0,
            init_temp: float = 0.,
            target_entropy: float | None = None,
            tau: float = 0.01,
            discount: float = 0.99,
            exp_prefix: str = ''
        ) -> None:
        super().__init__(name, log_dir)

        self.dummy_obs = dummy_obs
        self.dummy_action = dummy_action 
        self.obs_dim = dummy_obs.shape[-1]# FlattenObservationWrapper in default
        self.action_dim = dummy_action.shape[-1]
        self.auto_finetune_temp = auto_finetune_temp
        self.reward_scaling = reward_scaling
        self.tau = tau
        self.init_temp = init_temp 
        self.discount = discount 
        self.opt_kwargs = opt_kwargs
        # build trainer network
        self.double_critic = nn.vmap(
            MLPQCritic,
            in_axes = None, out_axes = 0,
            variable_axes = {'params': 0},  # Parameters are not shared between critics
            split_rngs = {'params': True},  # Different initializations
            axis_size = 2,  # Number of critics
        )(**qf_kwargs)
        self.actor = MLPTanhGaussianActor(self.action_dim, **actor_kwargs)
        # Temperature(alpha) model if necessary
        if self.auto_finetune_temp:
            self.temp = Temperature()
            # temp_params = self.temp.init(temp_rng)
            if target_entropy is None:
                self.target_entropy = -0.5 * self.action_dim
            else:
                self.target_entropy = target_entropy

    def init_trainer_state(self, rng):
        critic_rng, actor_rng, temp_rng = jax.random.split(rng, 3)
        # init AC model state
        critic_params = self.double_critic.init(critic_rng, self.dummy_obs, self.dummy_action) # 'in_axes' is None, so no need to infer 'axis_size' by input size  
        actor_params = self.actor.init(actor_rng, self.dummy_obs)
        critic_state = TargetTrainState.create(
            apply_fn=self.double_critic.apply,
            params=critic_params,
            target_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), critic_params),
            tx=self.set_optimizer(**self.opt_kwargs),
            n_updates=0,
        )
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=self.set_optimizer(**self.opt_kwargs)
        )
        # init temp model state
        if self.auto_finetune_temp:
            temp_params = self.temp.init(temp_rng)
            temp_state = TrainState.create(
                apply_fn=self.temp.apply,
                params=temp_params,
                tx=self.set_optimizer(**self.opt_kwargs)
            )
        else:
            temp_state = None
        return SACTrainerState(actor_state=actor_state, critic_state=critic_state, temp_state=temp_state, epoch_idx=0)


    def get_action(self, trainer_state, obs, rng, deterministic=False):
        """
        Return:
            action: jnp.Array with shape: (n, action_dim)
        """
        if not deterministic:
            # self.rng, rng = jax.random.split(self.rng)
            action = self.random_action(trainer_state.actor_state, obs, rng)
        else: 
            action = self.optimal_action(trainer_state.actor_state, obs)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def random_action(self, actor_state, obs, rng):
        """
        Args:
            actor_state: current 'actor_state'
            obs: jnp.Array with shape: (n, obs_dim)
            rng: random seed
        Return:
            action: jnp.Array with shape: (n, action_dim)
            rng: subkey for the next random seed 
        """
        action_dist = self.get_action_dist(actor_state, actor_state.params, obs)
        action = action_dist.sample(seed=rng)
        return action 

    @partial(jax.jit, static_argnames=['self'])
    def optimal_action(self, actor_state, obs):
        a_mean, _ = actor_state.apply_fn(actor_state.params, obs)
        action = jnp.tanh(a_mean)
        return action

    @partial(jax.jit, static_argnames=['self'])
    def update(self, trainer_state, batch_data, rng):
        """ implement an update of AC model once """
        """
        Args:
            batch_data: flashbax.sample.experience
        """
        rng_c, rng_a = jax.random.split(rng, 2)
        if self.auto_finetune_temp:
            alpha = trainer_state.temp_state.apply_fn(trainer_state.temp_state.params)
        else:
            alpha = self.init_temp

        critic_state, critic_loss = self.update_critic(
            trainer_state.critic_state, trainer_state.actor_state, alpha, batch_data, rng_c
        )        
        actor_state, entropy, actor_loss = self.update_actor(
            trainer_state.critic_state,
            trainer_state.actor_state, 
            alpha,
            batch_data,
            rng_a,
        )
        total_infos = {
            'tr/q_loss': critic_loss, 
            'tr/pi_loss': actor_loss
        }
        temp_state = None
        if self.auto_finetune_temp:
            temp_state, temp_loss = self.update_temp(trainer_state.temp_state, entropy)
            total_infos['tr/alpha_loss'] = temp_loss

        return SACTrainerState(actor_state=actor_state, critic_state=critic_state, temp_state=temp_state, epoch_idx=trainer_state.epoch_idx), total_infos

    def update_critic(self, critic_state, actor_state, alpha, batch_data, rng):
        next_obs = batch_data.second.obs # (bs, dim)
        rewards = batch_data.first.reward # (bs, )
        next_action, next_log_prob = self.sample_and_log_prob(
            actor_state, actor_state.params, next_obs, rng
        ) # shape: (bs, act_dim), (bs,)
        q_next = critic_state.apply_fn(critic_state.target_params, next_obs, next_action)  # shape: (2, bs, 1)
        min_q_next = jnp.min(q_next, axis=0) # (bs, 1)
        q_target = jax.lax.stop_gradient(rewards * self.reward_scaling + self.discount * (1. - batch_data.first.done) \
            * (min_q_next.squeeze() - alpha * next_log_prob)) # (bs,)
        def q_loss_fun(params):
            qs = critic_state.apply_fn(params, batch_data.first.obs, batch_data.first.action) #(2, bs, 1)
            loss = ((qs.squeeze() - q_target)**2).mean(axis=-1).sum()
            return loss
        critic_loss, critic_grad = jax.value_and_grad(q_loss_fun)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=critic_grad)
        critic_state = critic_state.replace(
            target_params=optax.incremental_update(
                critic_state.params, critic_state.target_params, self.tau
            ),
            n_updates=critic_state.n_updates+1
        )
        return critic_state, critic_loss 
    
    def update_actor(self, critic_state, actor_state, alpha, batch_data, rng):
        obs = batch_data.first.obs
        def actor_loss_fun(params):
            actions, log_prob = self.sample_and_log_prob(actor_state, params, obs, rng) 
            entropy = -log_prob.mean()
            qs = critic_state.apply_fn(critic_state.params, obs, actions) # (2, bs, 1)
            min_qs = jnp.min(qs, axis=0)
            loss = (alpha * log_prob - min_qs.squeeze()).mean()
            return loss, entropy
        (actor_loss, entropy), actor_grad = jax.value_and_grad(actor_loss_fun, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grad)
        return actor_state, entropy, actor_loss 

    def update_temp(self, temp_state, entropy):
        def temp_loss_fun(params):
            alpha = temp_state.apply_fn(params)
            loss = jnp.log(alpha) * jax.lax.stop_gradient((entropy - self.target_entropy).mean())
            return loss
        temp_loss, temp_grad = jax.value_and_grad(temp_loss_fun)(temp_state.params)
        temp_state = temp_state.apply_gradients(grads=temp_grad)
        return temp_state, temp_loss

    def get_action_dist(self, actor_state, params, obs):
        a_mean, a_log_std = actor_state.apply_fn(params, obs) # has clipped 'a_log_std'
        action_dist = distrax.Transformed(
            distrax.MultivariateNormalDiag(a_mean, jnp.exp(a_log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        return action_dist

    def sample_and_log_prob(self, actor_state, params, obs, rng):
        action_dist = self.get_action_dist(actor_state, params, obs)
        action, log_prob = action_dist.sample_and_log_prob(seed=rng)
        return action, log_prob 
    
    @property
    def null_total_infos(self):
        """ should match with the format of 'total_info' in func 'update()' """
        infos = {
            'tr/q_loss': jnp.array(0.0),
            'tr/pi_loss': jnp.array(0.0),
        }
        if self.auto_finetune_temp:
            infos['tr/alpha_loss'] = jnp.array(0.0)
        return infos

    def save_trainer_state(self, trainer_state_outs, outs_size):
        all_actor_params = trainer_state_outs.actor_state.params
        all_critic_params = trainer_state_outs.critic_state.params
        if self.auto_finetune_temp:
            all_temp_params = trainer_state_outs.temp_state.params
        
        for i in range(outs_size):
            model_params_dict = {
                'actor_params': jax.tree_util.tree_map(lambda x: x[i], all_actor_params),
                'critic_params': jax.tree_util.tree_map(lambda x: x[i], all_critic_params)
            }
            if self.auto_finetune_temp:
                model_params_dict['temp_params'] = jax.tree_util.tree_map(lambda x: x[i], all_temp_params)
            save_dir = os.path.join(self.log_dir, f'final_model_vmap_{i}')
            self.save_model(model_params_dict, save_dir)