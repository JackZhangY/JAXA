import multiprocessing
import os

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

def single_seed_run(command_format, gpu_idx, seed):
    os.system(command_format.format(gpu_idx, seed))


if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 4]
    gpu_idx = 0
    multiprocessing.set_start_method('spawn')

    # exclude 'NPROC=1/2/4' seems no influence on the speed?
    command_format = 'TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=1 TF_FORCE_GPU_ALLOW_GROWTH=false XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES={} python main_d4rl.py seed={}'
    for seed in seeds:
        p = multiprocessing.Process(target=single_seed_run, args=(command_format, gpu_idx, seed))
        p.start()
