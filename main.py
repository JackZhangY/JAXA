import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

import jax 
# jax.config.update('jax_default_device', jax.devices('gpu')[2]) # still a little memory leakage on other device
import hydra, omegaconf
import swanlab
from envs.wrappers import  NormalizeVecObservation, NormalizeVecReward
from envs.make_env import make_vec_env
import flashbax as fbx
from utils.misc import d4rl_to_fbx, omegaconf_to_dict
from agents import * 
from core import *


@hydra.main(config_path='configs', config_name='base.yaml')
def main(args):
    ### seed and device ###
    rng = jax.random.PRNGKey(args.seed)
    # device = jax.devices(backend=args.device.type)[int(args.device.idx)]
    # jax.default_device(device)

    ### save dir ###
    base_dir = args.logger.log_dir 
    # file directory of current experiment: e.g., 'base_dir/SAC/tau=0.001_seed=0'
    args.trainer.exp_prefix += f'_seed={args.seed}' # add the seed suffix
    exp_dir = os.path.join(base_dir, args.trainer.name, args.trainer.exp_prefix)
    if os.path.exists(exp_dir): # check if the experiment has been done 
        print('######## setting  have done ! ########')
        print(exp_dir)
        print('######################################')
        return None
    if args.swanlab.use:
        swanlab.init(
            project=args.swanlab.project,
            experiment_name='SAC(no norm)-seed={}'.format(args.seed),
            config=omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
            logdir=exp_dir
        )

    #### build Env & Replaybuffer ####
    if args.rl_paradigm == 'online':
        rng, rng_eval, rng_expl = jax.random.split(rng, 3)
        eval_env, eval_env_params, fake_trans = make_vec_env(args.env.name, rng_eval)
        expl_env, expl_env_params, _ = make_vec_env(args.env.name, rng_expl)
        if args.env.obs_r_norm:
            eval_env, expl_env = NormalizeVecObservation(eval_env), NormalizeVecObservation(expl_env)
            eval_env, expl_env = NormalizeVecReward(eval_env, args.trainer.discount), NormalizeVecReward(expl_env, args.trainer.discount)
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
        buffer_state = buffer.init(fake_trans)
    else: # offline rl # TODO: other RL paradigm, e.g., Meta RL, Offline Meta RL?
        # rng, rng_eval = jax.random.split(rng, 2)
        # eval_env, fake_trans = make_vec_env(args.env.name, args.env.norm, args.trainer.discount, rng_eval)
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
        buffer_state = buffer.init(fake_trans)
        buffer_state = d4rl_to_fbx(env_name=args.env.name, buffer_state=buffer_state, buffer=buffer)    

    ### build Trainer & RL algro ###
    rng, rng_tr, rng_rl = jax.random.split(rng, 3)
    if args.trainer.name  == 'SAC':
        trainer = SACTrainer(
            log_dir=exp_dir,
            rng=rng_tr,
            dummy_obs=fake_trans.obs,
            dummy_action=fake_trans.action,
            **args.trainer,
        )
    else:
        raise NotImplementedError('Not support current algorithm')

    # build RL algorithm
    if args.rl_paradigm == 'online':
        RLAlgo = OnlineRLParadigm(
            trainer=trainer,
            rng=rng_rl,
            exploration_env=(expl_env, expl_env_params),
            evaluation_env=(eval_env, eval_env_params),
            expl_env_nums=args.env.expl_num,
            eval_env_nums=args.env.eval_num,
            replay_buffer=(buffer, buffer_state),
            **args.rlalg,
        )
    else:
        raise NotImplementedError('Not suppoert current RL setting')

    # save config   
    trainer.logger.log_variant(os.path.join(exp_dir, 'configs.json'), omegaconf_to_dict(args))

    RLAlgo.train()

    # save model parameter
    if args.save_model:
        trainer.save_model(os.path.join(exp_dir, 'final_model_params')) 

    if args.swanlab.use:
        swanlab.finish()

if __name__ == '__main__':
    main()