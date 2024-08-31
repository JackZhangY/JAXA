import abc
from utils.misc import Transition
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from typing import Tuple, Any
from agents.base_trainer import BaseTrainer


class BaseParadigm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            args: Any,
            trainer: BaseTrainer,
            batch_size: int, 
            start_epoch: int,
            end_epoch: int,
            total_steps: int,
    ):
        self.args = args
        self.trainer = trainer
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.total_steps = total_steps

    @abc.abstractmethod
    def make_train(self):
        """ build 'train' function for different rl paradigm"""
        pass
    
    @abc.abstractmethod
    def epoch_train(self):
        """
        agent's epoch training.
        """
        pass
    
    @abc.abstractmethod
    def policy_evaluate(self):
        """
        policy evaluation function
        """
        pass