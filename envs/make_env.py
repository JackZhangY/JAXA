from envs.wrappers import (BraxGymnaxWrapper, FlattenObservationWrapper, LogWrapper, VecEnv, ClipAction)
from envs.d4rl_env import NormalizedBoxEnv
from envs.mp_vec_env import SubprocVecEnv
import gym
import jax
from typing import Any, Callable, Tuple, Union
from utils.misc import Transition
import chex
from omegaconf import DictConfig
from gymnax.environments import spaces


def make_vec_env(env_args: Any, rng: chex.PRNGKey):
    # env_type = env_args.name.split('-')[-1]
    env_type = env_args.type
    if env_type == 'gymnax':
        import gymnax
        env, env_params = gymnax.make(env_args.name)
        if env_args.obs_flat:
            env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        # construct a dummy Transition
        rng_obs, rng_act, rng_step = jax.random.split(rng, 3)
        _obs, _env_state = env.reset(rng_obs, env_params)
        if type(env.action_space()) == spaces.Discrete:
            action_dim = env.action_space().n
            _action = jax.random.randint(rng_act, (), minval=0, maxval=action_dim)
        ## TODO: elif type(env.action_space()) == spaces.Box 
        _, _, _reward, _done, _ = env.step(rng_step, _env_state, _action, env_params)
        _trans = Transition(obs=_obs, action=_action, reward=_reward, done=_done)
        obs_act_infos = [_trans, action_dim]
        env = VecEnv(env)
    elif env_type == 'Brax':
        env, env_params = BraxGymnaxWrapper(env_args.name, env_args.backend), None
        env = LogWrapper(env)
        env = ClipAction(env)
        # construct a dummy Transition
        rng_obs, rng_act, rng_step = jax.random.split(rng, 3)
        _obs, _env_state = env.reset(rng_obs, env_params)
        _action = env.action_space(env_params).sample(rng_act)
        _, _, _reward, _done, _ = env.step(rng_step, _env_state, _action, env_params)
        _trans = Transition(obs=_obs, action=_action, reward=_reward, done=_done)
        obs_act_infos = [_trans, None]
        env = VecEnv(env)
    # elif env_type == 'Envpool':
    #     # env = envpool.make_gymnasium(env_name, num_envs=env_nums, seed=int(rng[0])), None
    #     env, env_params = EnvpoolGymnaxWrapper(env_name, env_nums, int(rng[0]))
    else:
        raise NotImplementedError('Not support current env task')


    return env, env_params, obs_act_infos

def make_d4rl_vec_env(
    env_args: DictConfig,
    obs_stats: Tuple,
    seed: int = 0,
    start_method: Union[str, None] = None 
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_args: includes the env ID, optional obs normalization, optional max episode length
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :return: The wrapped environment
    """

    import d4rl
    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:

            if isinstance(env_args.name, str):
                env = gym.make(env_args.name)
                if env_args.obs_norm:
                    env = NormalizedBoxEnv(env)
                    env.set_obs_stats(obs_mean=obs_stats[0], obs_std=obs_stats[1])
                if env_args.max_episode_steps > 0:
                    env._max_episode_steps = env_args.max_episode_steps
            else:
                raise ValueError('Please input str type of env_id')

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env

        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(env_args.num_eval_envs)], start_method=start_method)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env