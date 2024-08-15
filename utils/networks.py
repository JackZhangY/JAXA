import abc

import jax
import jax.numpy as jnp
import flax.linen as nn

from flax.linen.initializers import variance_scaling, lecun_uniform, he_uniform, constant, zeros_init, orthogonal
from flax.core.frozen_dict import FrozenDict
from jax.typing import ArrayLike
from typing import Any, Callable, Sequence, Tuple, Dict
Initializer = Callable[[Any, Tuple[int, ...], Any], Any]


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
    
    def setup(self):
        layers = []
        for i in range(len(self.layer_dims)-1):
            layers.append(nn.Dense(self.layer_dims[i], kernel_init=self.kernel_init()))
            layers.append(activations[self.hidden_act])

        # add last layer
        last_kernel_init = self.kernel_init(self.last_w_scale) if self.last_w_scale > 0 else self.kernel_init()
        layers.append(nn.Dense(self.layer_dims[-1], kernel_init=last_kernel_init))
        # no activation for last layer by default
        if self.output_act != 'Linear':
            layers.append(activations[self.output_act])
        self.mlp = nn.Sequential(layers)

    def __call__(self, x):
        return self.mlp(x)

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
        obs = jnp.reshape((obs.shape[0], -1))  # ensure the flatten feature input
        q = self.mlp(obs)
        return q 

class ConvQNet(nn.Module):
    pass


class NAFQuadraticCritic(nn.Module):
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

    def __call__(self, obs, action):
        x = jnp.concat([obs, action], axis=-1) # (s,a) as input
        q = self.mlp(x)
        return q

class ConvQCritic(nn.Module):
    pass

##### Actor (Policy ) models ##########

### Note that 'Actor' models only output the mean (and std) of an action distribution, the true distribution will be built with 'distrax' Lib
class MLPGaussianActor(nn.Module):
    action_dim: int = 1
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': ['ReLU']})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0

    def setup(self):
        self.feature_net = MLP(
            layer_dims = list(self.hidden_cfg['hidden_dims']),
            hidden_act = self.hidden_cfg['hidden_act'],
            output_act= self.hidden_cfg['hidden_act'], # actually hidden_act for 'feature_net'
            kernel_init = self.kernel_init,
            last_w_scale= -1.0 # only contain hidden layer, not the final layer
        )
        self.action_mean = nn.Dense(self.action_dim, kernel_init=self.kernel_init(self.last_w_scale))
        self.action_std = nn.Sequential([
            nn.Dense(self.action_dim, kernel_init=self.kernel_init(self.last_w_scale)),
            activations['Sigmoid']
        ])        

    def __call__(self, obs):
        obs_feat = self.feature_net(obs)
        a_mean = self.action_mean(obs_feat) 
        a_std = self.action_std(obs_feat)
        return a_mean, a_std


class MLPTanhGaussianActor(nn.Module):
    action_dim: int = 1
    hidden_cfg: FrozenDict = FrozenDict({'hidden_dims': [64, 64], 'hidden_act': ['ReLU']})
    kernel_init: Initializer = default_init
    last_w_scale: float = -1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    def setup(self):
        self.actor_net = MLP(
            layer_dims=self.hidden_cfg['hidden_dims'] + [self.action_dim * 2],
            hidden_act=self.hidden_cfg['hidden_act'],
            kernel_init = self.kernel_init,
            last_w_scale= self.last_w_scale
        )

    def __call__(self, obs):
        a_mean, a_log_std = jnp.split(self.actor_net(obs), 2, axis=-1)
        a_log_std = jnp.clip(a_log_std, self.log_std_min, self.log_std_max)
        return a_mean, a_log_std




##### marco variable ####
VALUENETS = {
    'MLPQNet': MLPQNet,
    'MLPQCritic': MLPQCritic,
    'NAFQuadraticCritic': NAFQuadraticCritic
}

ACTORNETS = {
    'MLPGaussianActor': MLPGaussianActor,
    'MLPTanhGaussianActor': MLPTanhGaussianActor
}

        


























