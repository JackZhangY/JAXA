import jax.numpy as jnp
from typing import Dict
import chex
import jax

@chex.dataclass
class BaseExploration:
    epsilon_end: float = 0.9
    min_exploration_steps: int = 5e3

    def eps_threshold(self, step_counts):
        raise NotImplementedError('To be implemeneted')


@chex.dataclass
class EpsilonGreedy(BaseExploration):

    def eps_threshold(self, step_counts):
        cur_epsilon = jax.lax.cond(
            step_counts <= self.min_exploration_steps,
            lambda step_counts: 1.1,
            lambda step_counts: self.epsilon_end,
            step_counts,
        )
        return cur_epsilon 
        
@chex.dataclass
class LinearEpsilonGreedy(BaseExploration):
    """
    'epsilon_end' should be smaller than 'epsilon_start'
    """
    epsilon_start: float = 1.0 
    epsilon_steps: int = 1e5

    def eps_threshold(self, step_counts):
        cur_epsilon = jax.lax.cond(
            step_counts <= self.min_exploration_steps,
            lambda step_counts: 1.1,
            lambda step_counts: jnp.maximum(self.epsilon_start + step_counts * (self.epsilon_end - self.epsilon_start) / self.epsilon_steps, self.epsilon_end),
            step_counts,
        )
        return cur_epsilon 

@chex.dataclass
class ExponentialEpsilonGreedy(BaseExploration):
    epsilon_start: float = 1.0
    epsilon_decay: float = 1000.

    
    def eps_threshold(self, step_counts):
        cur_epsilon = jax.lax.cond(
            step_counts <= self.min_exploration_steps,
            lambda step_counts: 1.1,
            lambda step_counts: self.epsilon_end + (self.epsilon_start - self.epsilon_end) * jnp.exp(-1. * step_counts / self.epsilon_decay),
            step_counts,
        )
        return cur_epsilon 
        

def select_epsilon_exploration(expl_class='constant', expl_params={'episode_end': 0.9}):
    if expl_class == 'constant':
        return EpsilonGreedy(**expl_params)
    elif expl_class == 'linear':
        return LinearEpsilonGreedy(**expl_params)
    elif expl_class == 'exponential':
        return ExponentialEpsilonGreedy(**expl_params)
    else:
        raise ValueError('No such epsilon exploration')