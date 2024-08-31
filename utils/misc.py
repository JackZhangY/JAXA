from typing import Any
from enum import Enum
import chex, os, errno, json
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from omegaconf import DictConfig, ListConfig


##### misc definitions for logging #####
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def omegaconf_to_dict(oc):
    if isinstance(oc, DictConfig):
        d = {}
        for k, v in oc.items():
            d[k] = omegaconf_to_dict(v)
        return d
    if isinstance(oc, ListConfig):
        d = []
        for k in oc:
            d.append(omegaconf_to_dict(k))
        return d
    else:
        return oc

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


##### misc definitions for RL training #####
class TargetTrainState(TrainState):
    target_params: FrozenDict
    n_updates: int

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

@chex.dataclass
class CriticTrainerState:
    critic_state: TrainState
    epoch_idx: int

@chex.dataclass
class ACTrainerState:
    actor_state: TrainState
    critic_state: TrainState
    epoch_idx: int

@chex.dataclass
class SACTrainerState(ACTrainerState):
    temp_state: TrainState | None

def d4rl_to_fbx(env_name, buffer_state, buffer):
    return buffer_state 