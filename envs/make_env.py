from envs.wrappers import (BraxGymnaxWrapper, FlattenObservationWrapper, LogWrapper, VecEnv, ClipAction)
import gymnax, navix
import jax
from utils.misc import Transition
import chex


def make_vec_env(env_name: str, rng: chex.PRNGKey):
    env_type = env_name.split('-')[-1]
    if env_type == 'MinAtar':
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        # construct a dummy Transition
        rng_obs, rng_act, rng_step = jax.random.split(rng, 3)
        _obs, _env_state = env.reset(rng_obs, env_params)
        action_dim = env.action_space().n
        _action = jax.random.randint(rng_act, (), minval=0, maxval=action_dim)
        _, _, _reward, _done, _ = env.step(rng_step, _env_state, _action, env_params)
        _trans = Transition(obs=_obs, action=_action, reward=_reward, done=_done)
    elif env_type == 'Brax':
        env, env_params = BraxGymnaxWrapper(env_name.split('-')[0]), None
        env = LogWrapper(env)
        env = ClipAction(env)
        # construct a dummy Transition
        rng_obs, rng_act, rng_step = jax.random.split(rng, 3)
        _obs, _env_state = env.reset(rng_obs, env_params)
        _action = env.action_space(env_params).sample(rng_act)
        _, _, _reward, _done, _ = env.step(rng_step, _env_state, _action, env_params)
        _trans = Transition(obs=_obs, action=_action, reward=_reward, done=_done)
    else:
        raise NotImplementedError('Not support current env task')

    env = VecEnv(env)

    return env, env_params, _trans

