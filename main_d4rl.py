import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" #  if not this, warning OUT-OF-MEMORY for jax==0.3.xx
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"
# os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True" # if this, Fetal Error for jax==0.3.xx

import jax 
# jax.config.update('jax_default_device', jax.devices('gpu')[2]) # still a little memory leakage on other device
import hydra 
import wandb
import gym, d4rl
from envs.make_env import make_d4rl_vec_env
from utils.misc import omegaconf_to_dict
from replay_buffers.d4rl_buffer import make_d4rl_buffer
from agents import * 
from core.offline_rl_paradigm import OfflineRLParadigm

#####################################################
#######                                       #######
#######   Offline RL Algorithm Implementation ####### 
#######                                       #######
#####################################################
@hydra.main(config_path='configs', config_name='base_d4rl.yaml')
def main(args):
    print('starting......')
    ### seed and device ###
    rng = jax.random.PRNGKey(args.seed)
    base_dir = args.logger.log_dir 
    # file directory of current experiment: e.g., 'base_dir/SAC/tau=0.001_seed=0'
    exp_dir = os.path.join(base_dir, args.trainer.name, args.env.name, args.trainer.exp_prefix)
    if os.path.exists(exp_dir): # check if the experiment has been done 
        print('######## setting  have done ! ########')
        print(exp_dir)
        print('######################################')
        return None

    #### build Env & Replaybuffer ####

    print('before dummy env established')
    # build D4RL replay buffer
    init_env = gym.make(args.env.name)
    is_antmaze = ('antmaze' in args.env.name)
    buffer = make_d4rl_buffer(
        obs_normalize=args.env.obs_norm,
        reward_normalize=args.env.reward_norm,
        batch_size=args.rlalg.batch_size, 
        is_antmaze=is_antmaze
    )
    buffer_state, obs_mean, obs_std = buffer.init(init_env)

    obs_stats = (obs_mean, obs_std)
    eval_vec_env = make_d4rl_vec_env(args.env, obs_stats)
    dummy_obs = eval_vec_env.observation_space.sample()
    dummy_action = eval_vec_env.action_space.sample()
    
    ### build Trainer & RL algro ###
    if args.trainer.name  == 'IQL':
        trainer = IQLTrainer(
            log_dir=exp_dir,
            dummy_obs=dummy_obs,
            dummy_action=dummy_action,
            **args.trainer,
        )
    else:
        raise NotImplementedError('Not support current algorithm')

    # build RL algorithm
    RLAlgo = OfflineRLParadigm(
        args=args,
        trainer=trainer,
        evaluation_env=eval_vec_env,
        replay_buffer=(buffer, buffer_state),
        **args.rlalg,
    )

    # save config   
    trainer.logger.log_variant(os.path.join(exp_dir, 'configs.json'), omegaconf_to_dict(args))

    # implement training function
    train = RLAlgo.make_train(exp_dir=exp_dir, wandb_mode=args.wandb_mode)
    train_jit = jax.jit(train)
    outs = train_jit(rng=rng)

    eval_vec_env.close()

    if args.wandb_mode:
        wandb.finish()

    # if args.save_model:
        # trainer_state_outs = outs['trainer_state']
        # trainer.save_trainer_state(trainer_state_outs, args.num_seeds)

if __name__ == '__main__':
    main()