import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=34) 
matplotlib.rc('ytick', labelsize=34)
font = {'family' : 'normal',
        'size'   : 34}
import matplotlib.patches as mpatches

import numpy as np
import os

def main(num_tasks,
        datasets,
        algorithms,
        num_seeds,
        num_init_tasks,
        num_epochs,
        save_frequency,
        results_root,
        num_train_list):
    
    num_iter = num_epochs * (num_tasks - num_init_tasks + 1)
    x_axis = np.arange(0, num_iter, save_frequency)

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    name_order = {'ER Dynamic': 0,
                    'ER Compositional': 1,
                    'ER Joint': 2,
                    'ER Nocomponents': 3,
                    'EWC Dynamic': 4,
                    'EWC Compositional': 5,
                    'EWC Joint': 6,
                    'EWC Nocomponents': 7,
                    'VAN Dynamic': 8,
                    'VAN Compositional': 9,
                    'VAN Joint': 10,
                    'VAN Nocomponents': 11,
                    'FM Dynamic': 12,
                    'FM Compositional': 13} 
    version_map = {'Dynamic': 'Dyn. + Comp.',
                    'Compositional': 'Compositional',
                    'Joint': 'Joint',
                    'Nocomponents': 'No Comp.'}

    ylabel_map = {'acc': 'Accuracy', 'loss': 'Loss'}

    for dataset in datasets:
        final_vals = {}
        final_vals_all = {}
        final_errs_all = {}
        names = []
        for algorithm in algorithms:
            final_vals[algorithm] = {}
            for seed in range(num_seeds):
                iter_cnt = 0
                task_id = num_tasks - 1
                for idx_train, num_train in enumerate(num_train_list):
                    results_dir = os.path.join(results_root, dataset, algorithm, 'seed_{}'.format(seed), 'numtrain_{}'.format(num_train), 'task_{}'.format(task_id))
                    if 'dynamic' in algorithm and task_id >= num_init_tasks:
                        prev_results_dir = os.path.join(results_root, dataset, algorithm, 'seed_{}'.format(seed), 'numtrain_{}'.format(num_train), 'task_{}'.format(task_id - 1))
                        with open(os.path.join(prev_results_dir, 'num_components.txt')) as f:
                            line = f.readline()
                            prev_components = int(line.lstrip('final components: '))

                        with open(os.path.join(results_dir, 'num_components.txt')) as f:
                            line = f.readline()
                            curr_components = int(line.lstrip('final components: '))
                            keep_component = curr_components > prev_components
                            prev_components = curr_components
                    with open(os.path.join(results_dir, 'log.txt')) as f:
                        epoch = 0
                        while True:
                            if epoch == num_epochs:
                                break
                            if epoch == 0 or epoch == 1 or epoch % save_frequency == 0:  
                                try:
                                    next(f)    # epochs: 100, training task: 9
                                except StopIteration:
                                    print(dataset, algorithm, seed, task, epoch)
                                    raise
                                for task in range(task_id + 1):
                                    next(f)
                            epoch += 1

                        
                        ###### FINAL ###########
                        next(f)
                        for task in range(0, task_id + 1):
                            line = f.readline()
                            line = line.rstrip('\n')
                            i_0 = len('\ttask: {}\t'.format(task))
                            while i_0 != -1:
                                i_f  = line.find(':', i_0)
                                key = line[i_0 : i_f]
                                if key not in final_vals[algorithm]:
                                    final_vals[algorithm][key] = np.empty((num_seeds, num_tasks, len(num_train_list)))
                                if key not in final_vals_all:
                                    final_vals_all[key] = []
                                    final_errs_all[key] = []
                                i_0 = line.find(key + ': ', i_0) + len(key + ': ')
                                i_f = line.find('\t', i_0)
                                substr = line[i_0 : i_f] if i_f != 0 else line[i_0:]
                                try:
                                    val = float(substr)
                                except:
                                    if keep_component:
                                        val = float(substr.split(',')[0].lstrip('('))
                                    else:
                                        val = float(substr.split(',')[1].rstrip(')'))                                            

                                final_vals[algorithm][key][seed, task, idx_train] = val
                                i_0 = i_f if i_f == - 1 else i_f + 1
 

            for key in final_vals[algorithm]:
                final_vals_all[key].append(final_vals[algorithm][key].mean(axis=(0,1)))
                final_errs_all[key].append(final_vals[algorithm][key].mean(axis=1).std(axis=0))
            names.append(algorithm.split('_')[0].upper() + ' ' + algorithm.split('_')[1].title())

        idx = [x[0] for x in sorted(enumerate(names), key=lambda x:name_order[x[1]])]
        names = np.array(names)[idx]

        for key in final_vals_all:
            final_vals_all[key] = np.array(final_vals_all[key])[idx]
            final_errs_all[key] = np.array(final_errs_all[key])[idx] / np.sqrt(num_seeds)

        markers = ['o', '*', 'D', '^']
        names_tmp = [version_map[x.split(' ')[1]] for x in names]
        for key in final_vals_all:
            plt.figure(figsize=(10,10))
            plt.subplot(111)
            for vals, errs, m in zip(final_vals_all[key], final_errs_all[key], markers):
                if key == 'acc':
                    print(vals[-1])
                plt.plot(num_train_list, vals, linewidth=4, marker=m, markersize=16)
                plt.fill_between(num_train_list, vals - errs, vals + errs, alpha=0.5)
            plt.ylabel(ylabel_map[key], fontsize=34)
            plt.xlabel('# data points', fontsize=34)
            plt.title('Learning curves on ' + dataset, fontsize=34, y=1.1)
            plt.tight_layout()
            plt.savefig(os.path.join(results_root, dataset, key + '_limiteddata_{}.pdf'.format(dataset)), transparent=True) 
    print(names_tmp)
    legend = plt.legend(names_tmp, fontsize=34, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=4)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(results_root, 'limiteddata_legend.pdf'), dpi="figure", bbox_inches=bbox) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot accuracy vs number of training points for lifelong compositional learning')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=10, type=int)
    parser.add_argument('-d', '--datasets', dest='datasets', default='MNIST', 
            choices=['MNIST', 'Fashion', 'CIFAR', 'CUB', 'Omniglot',
                'Landmine', 'LondonSchool', 'FacialRecognition'],
            nargs='+')
    parser.add_argument('-alg', '--algorithms', dest='algos', default='er_compositional', 
        choices=['er_compositional', 'ewc_compositional', 'van_compositional',
                'er_joint', 'ewc_joint', 'van_joint',
                'er_nocomponents', 'ewc_nocomponents', 'van_nocomponents',
                'er_dynamic', 'ewc_dynamic', 'van_dynamic',
                'fm_compositional', 'fm_dynamic'],
                nargs='+')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=100, type=int)
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=1, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=4, type=int)
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=1, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default='./tmp/results')
    parser.add_argument('--num_train', dest='num_train_list', default=0, nargs='+')
    args = parser.parse_args()

    main(args.num_tasks,
        args.datasets,
        args.algos,
        args.num_seeds,
        args.num_init_tasks,
        args.num_epochs,
        args.save_frequency,
        args.results_root,
        args.num_train_list)