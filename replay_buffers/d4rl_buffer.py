import gym
from replay_buffers.base_buffer import BaseBuffer
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
import chex
from utils.misc import D4RLBufferState, OfflineTrans
from dataclasses import dataclass

@chex.dataclass(frozen=True)
class D4RLBuffer:
    init: Callable
    sample: Callable
    # TODO: add sample for offline to online RL setting
    # add: Callable  

def init(env, obs_normalize, reward_normalize, is_antmaze=False):

    import d4rl
    dataset = d4rl.qlearning_dataset(env)
    buffer_size = dataset['terminals'].shape[0]
    reward_scale = 1.0
    if reward_normalize:
        returns, lengths = [], []
        ep_ret, ep_len = 0., 0
        for r, d in zip(dataset['rewards'].squeeze(), dataset['terminals'].squeeze()):
            ep_ret += float(r)
            ep_len += 1
            if d or ep_len == env._max_episode_steps - 1:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0., 0
        lengths.append(ep_len)
        assert sum(lengths) == buffer_size, 'miscount number of offline data'
        reward_scale = env._max_episode_steps / (max(returns) - min(returns) + 1E-8)
        print('========= reward scale by traj returns: {}'.format(reward_scale))
        

    obs = jnp.array(dataset['observations'], dtype=jnp.float32)
    actions = jnp.array(dataset['actions'], dtype=jnp.float32)
    next_obs = jnp.array(dataset['next_observations'], dtype=jnp.float32)
    rewards = jnp.array(dataset['rewards'] * reward_scale, dtype=jnp.float32)
    dones = jnp.array(dataset['terminals']).astype(jnp.float32)

    obs_mean = None
    obs_std = None
    if obs_normalize:
        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-3
        obs = (obs - obs_mean) / obs_std
        next_obs = (next_obs - obs_mean) / obs_std
    if is_antmaze:
        rewards = rewards - 1.

    off_data = OfflineTrans(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones
    )
    buffer_state = D4RLBufferState(data=off_data, buffer_size=buffer_size)
    return buffer_state, obs_mean, obs_std
    
def sample(buffer_state: D4RLBufferState, rng: chex.PRNGKey, batch_size: chex.Numeric):
    batch_idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=buffer_state.buffer_size)
    batch_data = jax.tree_util.tree_map(lambda x: x[batch_idx], buffer_state.data)
    return batch_data

def make_d4rl_buffer(obs_normalize, reward_normalize, batch_size, is_antmaze):
    
    init_fn = partial(init, obs_normalize=obs_normalize, reward_normalize=reward_normalize, is_antmaze=is_antmaze)
    sample_fn = partial(sample, batch_size=batch_size)
    return D4RLBuffer(
        init=init_fn,
        sample=sample_fn
    )
