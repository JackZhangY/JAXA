import abc

import jax
import jax.numpy as jnp
import flax.linen as nn

from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, orthogonal
from flax.core.frozen_dict import FrozenDict
from typing import Any, Callable, Sequence, Tuple, Dict
Initializer = Callable[[Any, Tuple[int, ...], Any], Any]

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

activations = {
  'ReLU': nn.relu,
  'ELU': nn.elu,
  'Softplus': nn.softplus,
  'LeakyReLU': nn.leaky_relu,
  'Tanh': jnp.tanh,
  'Sigmoid': nn.sigmoid,
  'Exp': jnp.exp
}


def default_init(scale: float = jnp.sqrt(2)):
  return orthogonal(scale)

#############################################
## Basic networks for common RL algorithms ##
#############################################
class Temperature(nn.Module):
  """ Self-tuning temperature for SAC. """
  init_temp: float = 1.0

  def setup(self):
    self.log_temp = self.param('log_temp', init_fn=lambda seed: jnp.full((), jnp.log(self.init_temp)))

  def __call__(self):
    return jnp.exp(self.log_temp)

class MLP(nn.Module):
    """ Multilayer perceptron """
    layer_dims: Sequence[int]
    hidden_act: str = 'ReLU'
    output_act: str = 'Linear'
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0
    # layer_norm: bool = False
    # dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.layer_dims) - 1):
            x = nn.Dense(self.layer_dims[i], kernel_init=self.kernel_init())(x)
            x = activations[self.hidden_act](x)
            # if self.dropout_rate > 0.:
            #     x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        # last layer
        last_kernel_init = self.kernel_init(self.last_w_scale) if self.last_w_scale > 0 else self.kernel_init()
        x = nn.Dense(self.layer_dims[-1], kernel_init=last_kernel_init)(x)
        # no activation for last layer by default
        if self.output_act != 'Linear':
            x = activations[self.output_act](x)
        return x


###############################################
# Q value networks for value-based algorithms
###############################################
class MLPQNet(nn.Module):
    """ Q network built with MLP for dicrete action space --> Q(s) """
    action_dim: int = 2
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'ReLU'}) 
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0

    def setup(self):
        self.mlp = MLP(
            layer_dims = list(self.hidden_cfg['hidden_dims']) + [self.action_dim],
            hidden_act = self.hidden_cfg['hidden_act'],
            kernel_init = self.kernel_init,
            last_w_scale = self.last_w_scale 
        )

    def __call__(self, obs):
        obs = jnp.reshape(obs, (obs.shape[0], -1))  # ensure the flatten feature input
        q = self.mlp(obs)
        return q 

class ConvQNet(nn.Module):
    pass


###################################################
# Actor-Critic networks for policy-based algorithms
###################################################

#####  Critic models     #####
class MLPQCritic(nn.Module):
    """ Q critic network with MLP for continuous action space --> Q(s,a) """
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'ReLU'})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0

    def setup(self):
        self.mlp = MLP(
            layer_dims = list(self.hidden_cfg['hidden_dims']) + [1],
            hidden_act = self.hidden_cfg['hidden_act'],
            kernel_init = self.kernel_init,
            last_w_scale = self.last_w_scale
        )

    def __call__(self, obs_action):
        q = self.mlp(obs_action)
        return q

class MLPVCritic(nn.Module):
    """ V critic network with MLP for continuous action space --> V(s) """
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'ReLU'})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0

    def setup(self):
        self.mlp = MLP(
            layer_dims = list(self.hidden_cfg['hidden_dims']) + [1],
            hidden_act = self.hidden_cfg['hidden_act'],
            kernel_init = self.kernel_init,
            last_w_scale = self.last_w_scale
        )

    def __call__(self, obs):
        v = self.mlp(obs)
        return v 

class ConvQCritic(nn.Module):
    pass

##### Actor (Policy ) models ##########

### Note that 'Actor' models only output the mean (and std) of an action distribution, the true distribution will be built with 'distrax' Lib
class MLPGaussianActor(nn.Module):
    action_dim: int = 1
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': ['ReLU']})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0 
    # dropout_rate: float = 0.0

    def setup(self):
        self.action_mean = MLP(
            layer_dims = list(self.hidden_cfg['hidden_dims']) + [self.action_dim],
            hidden_act = self.hidden_cfg['hidden_act'],
            output_act= 'Tanh',
            kernel_init = self.kernel_init,
            last_w_scale= -1.0 
        )
        self.action_log_std = self.param('log_stds', nn.initializers.zeros, (self.action_dim, ))
        # self.action_mean = nn.Dense(self.action_dim, kernel_init=self.kernel_init(self.last_w_scale))
        # self.action_std = nn.Sequential([
        #     nn.Dense(self.action_dim, kernel_init=self.kernel_init(self.last_w_scale)),
        #     activations['Sigmoid']
        # ])        

    def __call__(self, obs):
        a_mean = self.action_mean(obs) # have activated by 'jnp.tanh()' in MLP
        a_log_std = jnp.clip(self.action_log_std, self.log_std_min, self.log_std_max)
        return a_mean, a_log_std


class MLPTanhGaussianActor(nn.Module):
    action_dim: int = 1
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': ['ReLU']})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    # dropout_rate: float = 0.0

    def setup(self):
        self.actor_net = MLP(
            layer_dims=list(self.hidden_cfg['hidden_dims']) + [self.action_dim * 2],
            hidden_act=self.hidden_cfg['hidden_act'],
            kernel_init = self.kernel_init,
            last_w_scale= self.last_w_scale,
            # dropout_rate = self.dropout_rate
        )

    def __call__(self, obs):
        a_mean, a_log_std = jnp.split(self.actor_net(obs), 2, axis=-1)
        a_log_std = jnp.clip(a_log_std, self.log_std_min, self.log_std_max)
        return a_mean, a_log_std

##### integrated AC model for NAF #####
class NAFQuadraticAC(nn.Module):
    action_dim: int = 1
    V_hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'ReLU'})
    L_hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'Tanh'})
    mu_hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': 'Tanh'})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0

    def setup(self):
        self.V_net = MLP(
            layer_dims=list(self.V_hidden_cfg['hidden_dims']) + [1],
            hidden_act=self.V_hidden_cfg['hidden_act'],
            kernel_init=self.kernel_init,
            last_w_scale=self.last_w_scale
        )
        self.mu_net = MLP(
            layer_dims=list(self.mu_hidden_cfg['hidden_dims']) + [self.action_dim],
            hidden_act=self.mu_hidden_cfg['hidden_act'],
            kernel_init=self.kernel_init,
            last_w_scale=self.last_w_scale
        )
        self.L_net = MLP( # lower triangle matrix net
            layer_dims=list(self.L_hidden_cfg['hidden_dims']) + [self.action_dim * (self.action_dim + 1) // 2],
            hidden_act=self.L_hidden_cfg['hidden_act'],
            kernel_init=self.kernel_init,
            last_w_scale=self.last_w_scale
        )
    
    def __call__(self, obs, action):
        V = self.V_net(obs) # (bs, 1)
        mu = self.mu_net(obs) # (bs, act_dim)
        L = jax.vmap(self.vector_to_lower_triangular)(self.L_net(obs))
        L_T = jnp.transpose(L, axes=(0,2,1)) 
        P = L @ L_T # (bs, act_dim, act_dim)
        a_mu = jnp.expand_dims(action-mu, axis=-1) # (bs, act_dim, 1)
        a_mu_T = jnp.transpose(a_mu, axes=(0,2,1))
        A = -0.5 * (a_mu_T @ P @ a_mu).squeeze(axis=-1)
        Q = V + A
        return Q

    def get_mu(self, obs):
        return self.mu_net(obs)
    
    def get_v(self, obs):
        return self.V_net(obs)
    
    def vector_to_lower_triangular(self, v):
        # Transform the vector into a lower triangular matrix
        m = jnp.zeros((self.action_size, self.action_size), dtype=v.dtype).at[jnp.tril_indices(self.action_size)].set(v)
        # Apply exp to get non-negative diagonal
        m = m.at[jnp.diag_indices(self.action_size)].set(jnp.exp(jnp.diag(m)))
        return m

##### marco variable ####
# VALUENETS = {
#     'MLPQNet': MLPQNet,
#     'MLPQCritic': MLPQCritic,
# }

# ACTORNETS = {
#     'MLPGaussianActor': MLPGaussianActor,
#     'MLPTanhGaussianActor': MLPTanhGaussianActor
# }

        


























