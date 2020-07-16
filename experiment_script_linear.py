import subprocess
import os
import sys
import torch
from itertools import cycle

algorithms = ['er_compositional', 'ewc_compositional', 'van_compositional']
algorithms += ['er_joint', 'ewc_joint', 'van_joint']
algorithms += ['er_nocomponents', 'ewc_nocomponents', 'van_nocomponents']
datasets = ['Landmine', 'FacialRecognition', 'LondonSchool']
num_epochs = 10000
update_frequency = 10000
save_frequency = 100
init_mode = 'one_module_per_task'
results_root = 'results/linear'

processes = set()
did_not_start = 0
did_not_finish = 0
finished = 0
architecture = 'linear'
ewc_lambda = 1e-3
replay_size = 10

num_cpu = 20

for i in range(10):
    for d in datasets:
        if d == 'Landmine':
            num_tasks = 29
            num_columns = 4
        elif d == 'FacialRecognition':
            num_tasks = 21
            num_columns = 4
        elif d == 'LondonSchool':
            num_tasks = 139
            num_columns = 4
        init_tasks = num_columns
        for a in algorithms:
            while len(processes) == num_cpu:
                for p in cycle(processes):
                    try:
                        p.wait(1)
                        processes.remove(p)
                        break
                    except subprocess.TimeoutExpired:
                        pass

            results_path = os.path.join(results_root, d, a, 'seed_{}'.format(i))
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
            my_env['CUDA_VISIBLE_DEVICES'] = ''
            args = ['python', 'lifelong_experiment_linear.py',
                '-T', str(num_tasks), 
                '-d', d,
                '-e', str(num_epochs),
                '-f', str(update_frequency),
                '--lambda', str(ewc_lambda),
                '--replay', str(replay_size),
                '-l', str(num_columns),
                '-k', str(init_tasks),
                '-i', init_mode,
                '-arc', architecture,
                '-alg', a,
                '-n', str(1),
                '-r', results_root,
                '-sf', str(save_frequency),
                '--initial_seed', str(i)]
            p = subprocess.Popen(args, env=my_env)
            processes.add(p)

print(did_not_start, did_not_finish, finished)


