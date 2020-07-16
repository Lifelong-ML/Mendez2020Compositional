import subprocess
import os
import sys
import torch
from itertools import cycle
import numpy as np

algorithms = ['er_compositional', 'er_joint', 'er_nocomponents', 'er_dynamic']
datasets = ['MNIST', 'Fashion', 'CUB']
num_epochs = 1000
mini_batch = 32
update_frequency = 10
save_frequency = num_epochs + 1     # never
init_mode = 'random_onehot'
results_root = 'results/limiteddata'

num_gpus = torch.cuda.device_count()

gpu_use_total = np.zeros(num_gpus)
cuda_device_dict = {}
counter = 0
process_gpu_use = {}
did_not_start = 0
did_not_finish = 0
finished = 0
for i in range(50):
    for d in datasets:
        if d ==  'MNIST':
            num_tasks = 10
            size = 64
            num_layers = 4
            init_tasks = 4
            architecture = 'mlp'
            gpu_use = 20
            num_train_list = [16, 32, 64, 128, 256, 512]
        elif d ==  'Fashion':
            num_tasks = 10
            size = 64
            num_layers = 4
            init_tasks = 4
            architecture = 'mlp'
            gpu_use = 20
            num_train_list = [16, 32, 64, 128, 256, 512]
        elif d ==  'CIFAR':
            num_tasks = 20
            size = 50
            num_layers = 4
            init_tasks = 4
            architecture = 'cnn'
            gpu_use = 25
        elif d ==  'CUB':
            num_tasks = 20
            size = 256
            num_layers = 4
            init_tasks = 4
            architecture = 'mlp'
            gpu_use = 20
            num_train_list = [4, 8, 16, 32, 64, 128]
        elif d ==  'Omniglot':
            num_tasks = 50
            size = 53
            num_layers = 4
            init_tasks = 4
            architecture = 'cnn'
            gpu_use = 25
            
        for a in algorithms:
            for num_train in num_train_list:
                ewc_lambda = 1e-3
                cuda_device = counter % num_gpus
                while np.all(gpu_use_total + gpu_use > 100):
                    for p in cycle(process_gpu_use):
                        try:
                            p.wait(1)
                            gpu_use_remove = process_gpu_use[p]
                            gpu_use_total[cuda_device_dict[p]] -= gpu_use_remove
                            del process_gpu_use[p]
                            del cuda_device_dict[p]
                            break
                        except subprocess.TimeoutExpired:
                            pass

                cuda_device = np.argmin(gpu_use_total)
                results_path = os.path.join(results_root, d, a, 'seed_{}'.format(i), 'numtrain_{}'.format(num_train))
                print(results_path + ': ', end='')
                if not os.path.isdir(results_path):
                    print('Did not start')
                    did_not_start += 1
                else:
                    completed_tasks = len([name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))])
                    if completed_tasks != num_tasks:
                        print('Did not finish', end='')
                        did_not_finish += 1
                    else:
                        print('Finished')
                        finished += 1
                        continue
                my_env = os.environ.copy()
                my_env['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
                args = ['python', 'lifelong_experiment.py',
                    '-T', str(num_tasks), 
                    '-d', d,
                    '-e', str(num_epochs),
                    '-b', str(mini_batch),
                    '-f', str(update_frequency),
                    '--lambda', str(ewc_lambda),
                    '--replay', str(min(mini_batch, max(num_train // 10, 1))),
                    '-s', str(size),
                    '-l', str(num_layers),
                    '-k', str(init_tasks),
                    '-i', init_mode,
                    '-arc', architecture,
                    '-alg', a,
                    '-n', str(1),
                    '-r', results_root,
                    '-sf', str(save_frequency),
                    '--initial_seed', str(i),
                    '--num_train', str(num_train)]
                p = subprocess.Popen(args, env=my_env)
                process_gpu_use[p] = gpu_use
                gpu_use_total[cuda_device] += gpu_use
                counter += 1
                cuda_device_dict[p] = cuda_device
                print(cuda_device)

print(did_not_start, did_not_finish, finished)


