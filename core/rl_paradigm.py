import abc
from collections import OrderedDict
import gtimer as gt
import flashbax as fbx
import jax
from typing import Tuple, Any
import chex
from agents.base_trainer import BaseTrainer


class BaseParadigm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer: BaseTrainer,
            rng: chex.PRNGKey,
            exploration_env: Tuple[Any, Any] | None,
            evaluation_env: Tuple[Any, Any],
            expl_env_nums: int,
            eval_env_nums: int,
            start_epoch: int,
            replay_buffer: Tuple[Any, Any],
    ):
        self.trainer = trainer
        self.rng = rng
        self.expl_env_nums = expl_env_nums
        self.eval_env_nums = eval_env_nums
        if exploration_env is not None:
            self.expl_env, self.expl_env_params = exploration_env[0], exploration_env[1]
        self.eval_env, self.eval_env_params = evaluation_env[0], evaluation_env[1]
        self.buffer, self.buffer_state = replay_buffer[0], replay_buffer[1]
        self.start_epoch = start_epoch 
        self.total_steps = 0


    def _get_epoch_timings(self):
        times_itrs = gt.get_times().stamps.itrs
        times = OrderedDict()
        epoch_time = 0
        for key in sorted(times_itrs):
            time = times_itrs[key][-1]
            epoch_time += time
            times['time/{} (s)'.format(key)] = time
        times['time/epoch (s)'] = epoch_time
        times['time/total (s)'] = gt.get_times().total
        return times

    @abc.abstractmethod
    def train(self):
        """ Different implements for different rl paradigm"""
        pass

    def on_train(self):
        """
        online agent training.
        """

        raise NotImplementedError('on_train must implemented by inherited class for online setting')

    def off_train(self):
        """
        offline agent training 
        """

        raise NotImplementedError('off_train must implemented by inherited class for offline setting')
    