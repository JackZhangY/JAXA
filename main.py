import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import jax 
# jax.config.update('jax_default_device', jax.devices('gpu')[2]) # still a little memory leakage on other device
import hydra 
import wandb
from envs.wrappers import  NormalizeVecObservation, EvalNormalizeVecObservation
from envs.make_env import make_vec_env
import flashbax as fbx
from utils.misc import d4rl_to_fbx, omegaconf_to_dict
from agents import * 
from core import *

@hydra.main(config_path='configs', config_name='base.yaml')
def main(args):
    ### seed and device ###
    rng = jax.random.PRNGKey(args.seed)
    ### save dir ###
    base_dir = args.logger.log_dir 
    # file directory of current experiment: e.g., 'base_dir/SAC/tau=0.001_seed=0'
    exp_dir = os.path.join(base_dir, args.trainer.name, args.env.name, args.trainer.exp_prefix)
    if os.path.exists(exp_dir): # check if the experiment has been done 
        print('######## setting  have done ! ########')
        print(exp_dir)
        print('######################################')
        return None

    #### build Env & Replaybuffer ####
    if args.rl_paradigm == 'online':
        rng, rng_eval, rng_expl = jax.random.split(rng, 3)
        eval_env, eval_env_params, fake_trans = make_vec_env(args.env, rng_eval)
        expl_env, expl_env_params, _ = make_vec_env(args.env, rng_expl)
        if args.env.obs_norm:
            eval_env, expl_env = EvalNormalizeVecObservation(eval_env), NormalizeVecObservation(expl_env)
        # build ReplayBuffer
        buffer = fbx.make_flat_buffer(
            max_length=args.buffer.max_replay_buffer_size,
            min_length=args.rlalg.batch_size,
            sample_batch_size=args.rlalg.batch_size,
            add_sequences=False,
            add_batch_size=args.env.expl_num
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample)
        )
    else: # offline rl # TODO: other RL paradigm, e.g., Meta RL, Offline Meta RL?
        # TODO: init D4RL env:
        # build ReplayBuffer
        buffer = fbx.make_flat_buffer(
            max_length=args.buffer.max_replay_buffer_size,
            min_length=args.rlalg.batch_size,
            sample_batch_size=args.rlalg.batch_size
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample)
        )
        buffer_state = d4rl_to_fbx(env_name=args.env.name, buffer_state=buffer_state, buffer=buffer)    

    ### build Trainer & RL algro ###
    if args.trainer.name  == 'SAC':
        trainer = SACTrainer(
            log_dir=exp_dir,
            dummy_obs=fake_trans.obs,
            dummy_action=fake_trans.action,
            **args.trainer,
        )
    else:
        raise NotImplementedError('Not support current algorithm')

    # build RL algorithm
    if args.rl_paradigm == 'online':
        RLAlgo = OnlineRLParadigm(
            args=args,
            trainer=trainer,
            exploration_env=(expl_env, expl_env_params),
            evaluation_env=(eval_env, eval_env_params),
            expl_env_nums=args.env.expl_num,
            eval_env_nums=args.env.eval_num,
            replay_buffer=(buffer, fake_trans),
            **args.rlalg,
        )
    else:
        raise NotImplementedError('Not suppoert current RL setting')

    # save config   
    trainer.logger.log_variant(os.path.join(exp_dir, 'configs.json'), omegaconf_to_dict(args))

    # implement training function
    rng_seeds = jax.random.split(rng, args.num_seeds)
    pseudo_seeds= jax.numpy.arange(args.num_seeds)
    train_vjit = jax.jit(jax.vmap(RLAlgo.make_train(exp_dir=exp_dir, wandb_mode=args.wandb_mode)))
    outs = train_vjit(rng_seeds, pseudo_seeds)
    if args.wandb_mode:
        wandb.finish()

    if args.save_model:
        trainer_state_outs = outs['trainer_state']
        trainer.save_trainer_state(trainer_state_outs, args.num_seeds)

if __name__ == '__main__':
    main()