from functools import partial
from agents.base_trainer import BaseTrainer
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax, chex, optax
import jax.numpy as jnp
import distrax
from utils.misc import TrainState, TargetTrainState, IQLTrainerState
from typing import Dict
from utils.networks import *




class IQLTrainer(BaseTrainer):
    def __init__(
            self,
            name: str, 
            log_dir: str,
            qf_kwargs: FrozenDict,
            vf_kwargs: FrozenDict,
            actor_kwargs: FrozenDict,
            dummy_obs: chex.Array,
            dummy_action: chex.Array,
            critic_opt_kwargs: Dict,
            actor_opt_kwargs: Dict,
            reward_scaling: float = 1.0,
            reward_bias: float = 0.0,
            tau: float = 0.01,
            discount: float = 0.99,
            clip_score: float = jnp.inf,
            quantile: float = 0.5,
            beta: float = 1.0,
            exp_prefix: str = ''
        ):
        super().__init__(name, log_dir)

        self.dummy_obs = dummy_obs
        self.dummy_action = dummy_action
        self.obs_dim = dummy_obs.shape[-1]# FlattenObservationWrapper in default
        self.action_dim = dummy_action.shape[-1]

        self.critic_opt_kwargs = critic_opt_kwargs
        self.actor_opt_kwargs = actor_opt_kwargs
        self.discount = discount
        self.tau = tau 
        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.clip_score = clip_score
        self.quantile = quantile
        self.beta = beta

        # build trainer network
        self.double_Q_critic = nn.vmap(
            MLPQCritic,
            in_axes = None, out_axes = 0,
            variable_axes = {'params': 0},  # Parameters are not shared between critics
            split_rngs = {'params': True},  # Different initializations
            axis_size = 2,  # Number of critics
        )(**qf_kwargs)
        self.V_critic = MLPVCritic(**vf_kwargs)
        self.actor = MLPGaussianActor(self.action_dim, **actor_kwargs)

    def init_trainer_state(self, rng):
        rng_q, rng_v, rng_actor = jax.random.split(rng, 3)
        # init AC model state
        dummy_obs_action = jnp.concatenate([self.dummy_obs, self.dummy_action], axis=-1)
        q_critic_params = self.double_Q_critic.init(rng_q, dummy_obs_action) # 'in_axes' is None, so no need to infer 'axis_size' by input size  
        v_critic_params = self.V_critic.init(rng_v, self.dummy_obs)
        actor_params = self.actor.init(rng_actor, self.dummy_obs)
        q_critic_state = TargetTrainState.create(
            apply_fn=self.double_Q_critic.apply,
            params=q_critic_params,
            target_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), q_critic_params),
            tx=self.set_optimizer(**self.critic_opt_kwargs),
            n_updates=jnp.array(0.),
        )
        v_critic_state =TrainState.create(
            apply_fn=self.V_critic.apply,
            params=v_critic_params,
            tx=self.set_optimizer(**self.critic_opt_kwargs)
        )
        actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=self.set_optimizer(**self.actor_opt_kwargs)
        )

        return IQLTrainerState(
            actor_state=actor_state, 
            q_critic_state=q_critic_state, 
            v_critic_state=v_critic_state, 
            epoch_idx=jnp.array(0.))
    
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
        action, _ = actor_state.apply_fn(actor_state.params, obs) # has activated by 'jnp.tanh'
        return action

    def update(self, batch_data, trainer_state, rng):

        new_v_critic_state, v_loss = self.update_V_critic(
            trainer_state.q_critic_state, trainer_state.v_critic_state, batch_data
        )

        new_actor_state, actor_loss = self.update_actor(
            trainer_state.q_critic_state, new_v_critic_state, 
            trainer_state.actor_state, batch_data, rng
        )

        new_q_critic_state, q_loss = self.update_Q_critic(
            trainer_state.q_critic_state, new_v_critic_state, batch_data
        )
        total_infos = {
            'tr/q_loss': q_loss, 
            'tr/v_loss': v_loss, 
            'tr/pi_loss': actor_loss
        }

        return IQLTrainerState(
            actor_state=new_actor_state, 
            q_critic_state=new_q_critic_state, 
            v_critic_state=new_v_critic_state, 
            epoch_idx=trainer_state.epoch_idx), total_infos

    def update_Q_critic(self, q_critic_state, v_critic_state, batch_data):
        next_obs = batch_data.next_obs
        rewards = self.reward_scaling * batch_data.rewards+ self.reward_bias # (bs,)
        dones = batch_data.dones #(bs,)

        v_next = v_critic_state.apply_fn(v_critic_state.params, next_obs) #(bs, 1)
        q_target = jax.lax.stop_gradient(rewards + self.discount * (1. - dones) * v_next.squeeze()) #(bs,)

        def q_loss_fun(params): 
            obs_actions = jnp.concatenate([batch_data.obs, batch_data.actions], axis=-1) # for jax==0.3.xx ('d4rlax')
            qs = q_critic_state.apply_fn(params,obs_actions) #(2, bs, 1)
            loss = ((qs.squeeze() - q_target)**2).mean(axis=-1).sum()
            return loss
        
        q_critic_loss, q_critic_grad = jax.value_and_grad(q_loss_fun)(q_critic_state.params)
        q_critic_state = q_critic_state.apply_gradients(grads=q_critic_grad)
        q_critic_state = q_critic_state.replace(
            target_params=optax.incremental_update(
                q_critic_state.params, q_critic_state.target_params, self.tau
            ),
            n_updates=q_critic_state.n_updates+1
        )
        return q_critic_state, q_critic_loss 

    def update_V_critic(self, q_critic_state, v_critic_state, batch_data):
        obs = batch_data.obs
        actions = batch_data.actions
        obs_actions = jnp.concatenate([obs, actions], axis=-1) # for jax==0.3.xx ('d4rlax')
        q_cur = q_critic_state.apply_fn(q_critic_state.target_params, obs_actions)
        min_q_cur = jax.lax.stop_gradient(jnp.min(q_cur, axis=0)) #(bs, 1)

        def v_loss_fun(params):
            vs = v_critic_state.apply_fn(params, obs) #(bs, 1) 
            qv_diff = min_q_cur - vs # (bs, 1)
            weight = jnp.where(qv_diff > 0, self.quantile, (1 - self.quantile))
            loss = (weight * (qv_diff ** 2)).mean()
            return loss 

        v_critic_loss, v_critic_grad = jax.value_and_grad(v_loss_fun)(v_critic_state.params)
        v_critic_state = v_critic_state.apply_gradients(grads=v_critic_grad)

        return v_critic_state, v_critic_loss

    def update_actor(self, q_critic_state, v_critic_state, actor_state, batch_data, rng):
        obs = batch_data.obs
        actions = batch_data.actions
        obs_actions = jnp.concatenate([obs, actions], axis=-1) # for jax==0.3.xx ('d4rlax')
        target_q = q_critic_state.apply_fn(q_critic_state.target_params, obs_actions)
        min_target_q = jnp.min(target_q, axis=0) #(bs, 1)
        vs = v_critic_state.apply_fn(v_critic_state.params, obs) #(bs, 1) 
        adv = (min_target_q - vs).squeeze() #(bs,)
        exp_adv = jax.lax.stop_gradient(
            jnp.minimum(jnp.exp(adv * self.beta), self.clip_score)
        )

        def actor_loss_fun(params):
            action_dist = self.get_action_dist(actor_state, params, obs)
            a_log_prob = action_dist.log_prob(actions) #(bs,)
            actor_loss = -(a_log_prob * exp_adv).mean()
            return actor_loss
        
        actor_loss, actor_grad = jax.value_and_grad(actor_loss_fun)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=actor_grad)
        
        return actor_state, actor_loss

    def get_action_dist(self, actor_state, params, obs):
        a_mean, a_log_std = actor_state.apply_fn(params, obs) # has clipped 'a_log_std'
        action_dist = distrax.MultivariateNormalDiag(a_mean, jnp.exp(a_log_std)) # 'a_mean' has activated by 'jnp.tanh()' 
        return action_dist
    

    @property
    def null_total_infos(self):
        """ should match with the format of 'total_infos' in func 'update()' """
        infos = {
            'tr/q_loss': jnp.array(0.0),
            'tr/v_loss': jnp.array(0.0),
            'tr/pi_loss': jnp.array(0.0)
        }
        return infos