from utils.logger import Logger 
import abc
from typing import Union, Callable, Any, Dict
import jax, optax, chex
from flax.core.frozen_dict import FrozenDict 
# from jax._src.typing import Array, ArrayLike
from omegaconf import DictConfig, ListConfig
import orbax.checkpoint
from flax.training import orbax_utils


class BaseTrainer(object, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            name: str, 
            log_dir: str, 
            # rng: chex.PRNGKey
        ):
        self.name = name
        self.logger = Logger(log_dir)
        # self.rng = rng

    def set_optimizer(
            self,
            opt_name: str = 'adam',
            grad_clip: float = -1.,
            max_grad_norm: float = -1.,
            anneal: Union[Dict, None] = None,
            lr_kwargs: Dict = {'learning_rate': 3e-4} 
    ):
        if anneal is not None:
            sche_name = anneal['name']
            del anneal['name']
            lr_kwargs['learning_rate'] = getattr(optax, sche_name.lower())(**anneal)

        assert not (grad_clip > 0 and max_grad_norm > 0), 'Cannot apply both grad_clip and max_grad_norm at the same time'
        if grad_clip > 0:
            optimizer = optax.chain(
                optax.clip(grad_clip),
                getattr(optax, opt_name.lower())(**lr_kwargs)
            )
        elif max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                getattr(optax, opt_name.lower())(**lr_kwargs)
            )
        else:
            optimizer = getattr(optax, opt_name.lower())(**lr_kwargs)
        return optimizer


    @abc.abstractmethod
    def update(self):
        pass


    @abc.abstractmethod
    def get_action(self, actor_state: Any, obs: chex.Array, deterministic=False):
        pass
    
    @property
    def model_params(self,):
        """ a Dict includes all model parameters"""
        raise NotImplementedError

    def save_model(self, save_dir):
        orbax_ckpt = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(self.model_params)
        orbax_ckpt.save(save_dir, self.model_params, save_args=save_args)
