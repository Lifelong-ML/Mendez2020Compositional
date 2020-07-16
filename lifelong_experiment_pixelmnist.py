'''
Notes
    - Initialization should be in batch, to 
    avoid forgetting on the early ones and
    modules actually being trained for 
    reusability
    - Potentially add a schedule for the 
    component updates or make the updates 
    soft (like in soft DQN updates)
    - 
'''

import struct
import numpy as np
import torch
import argparse
import os
import matplotlib.pyplot as plt

from datasets import datasets

from models.mlp import MLP
from models.mlp_soft_lifelong_pixelmnist import MLPSoftLL

# Explicitly compositional with dynamic module number (Ours)
from learners.er_dynamic import CompositionalDynamicER
from learners.ewc_dynamic import CompositionalDynamicEWC
from learners.van_dynamic import CompositionalDynamicVAN
from learners.fm_dynamic import CompositionalDynamicFM

# Explicitly compositional (Ours)
from learners.er_compositional import CompositionalER
from learners.van_compositional import CompositionalVAN
from learners.ewc_compositional import CompositionalEWC
from learners.fm_compositional import CompositionalFM

# Implicitly compositional baselines (composition in the model, not in training)
from learners.ewc_joint import JointEWC
from learners.er_joint import JointER
from learners.van_joint import JointVAN

# No-components baselines (no composition in the model or in training)
from learners.ewc_nocomponents import NoComponentsEWC
from learners.er_nocomponents import NoComponentsER
from learners.van_nocomponents import NoComponentsVAN

SEED_SCALE = 10

def main(num_tasks=10,
        num_epochs=100,
        batch_size=64,
        component_update_frequency=100,
        ewc_lambda=1e-5,
        replay_size=-1,
        layer_size=64,
        num_layers=4,
        num_init_tasks=4,
        init_mode='random_onehot',
        algorithm='er_compositional',
        num_seeds=1,
        results_root='./tmp/results',
        save_frequency=1,
        initial_seed=0,
        num_train=-1):
    '''
    TODOS:
    Add module addition step
    '''
    # raise ValueError('Read TODO above for next steps')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seed in range(initial_seed, initial_seed + num_seeds):
        torch.manual_seed(seed * SEED_SCALE)
        np.random.seed(seed * SEED_SCALE)
        
        torch_dataset = datasets.MNISTPixels(num_tasks)
        
        net = MLPSoftLL(torch_dataset.features,
                                size=layer_size,
                                depth=num_layers,
                                num_classes=torch_dataset.num_classes, 
                                num_tasks=num_tasks,
                                num_init_tasks=num_init_tasks,
                                init_ordering_mode=init_mode,
                                device=device)
        batch_size = torch_dataset.max_batch_size
            
        
        net.train()     # training mode
        kwargs = {}
        results_dir=os.path.join(results_root, 'MNISTPixels', algorithm, 'seed_{}'.format(seed))
        
        if algorithm == 'er_compositional':
            if replay_size == -1:
                replay_size = batch_size
            agent = CompositionalER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_compositional':
            agent = CompositionalEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'van_compositional':
            agent = CompositionalVAN(net, results_dir=results_dir)
        elif algorithm == 'fm_compositional':
            agent = CompositionalFM(net, results_dir=results_dir)
        elif algorithm == 'er_joint':
            if replay_size == -1:
                replay_size = batch_size
            agent = JointER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_joint':
            agent = JointEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'van_joint':
            agent = JointVAN(net, results_dir=results_dir)
        elif algorithm == 'er_nocomponents':
            if replay_size == -1:
                replay_size = batch_size
            agent = NoComponentsER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_nocomponents':
            agent = NoComponentsEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'van_nocomponents':
            agent = NoComponentsVAN(net, results_dir=results_dir)
        elif algorithm == 'er_dynamic':
            if replay_size == -1:
                replay_size = batch_size
            agent = CompositionalDynamicER(net, replay_size, results_dir=results_dir)
        elif algorithm == 'ewc_dynamic':
            agent = CompositionalDynamicEWC(net, ewc_lambda, results_dir=results_dir)
        elif algorithm == 'van_dynamic':
            agent = CompositionalDynamicVAN(net, results_dir=results_dir)
        elif algorithm == 'fm_dynamic':
            agent = CompositionalDynamicFM(net, results_dir=results_dir)
        else:
            raise NotImplementedError('{} algorithm is not supported'.format(algorithm))

        for task_id, trainset in enumerate(torch_dataset.trainset):
            trainloader = (
                torch.utils.data.DataLoader(trainset,
                    # batch_size=batch_size,
                    batch_size=len(trainset),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    ))
            testloaders = {task: torch.utils.data.DataLoader(testset,
                                # batch_size=torch_dataset.max_batch_size,
                                batch_size=len(testset),
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                ) for task, testset in enumerate(torch_dataset.testset[:(task_id+1)])}

            if 'dynamic' in algorithm:
                valloader = torch.utils.data.DataLoader(torch_dataset.valset[task_id],
                                batch_size=torch_dataset.max_batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                )
                kwargs = {'valloader': valloader}

            agent.train(trainloader, 
                task_id, 
                component_update_freq=component_update_frequency,
                num_epochs=num_epochs,
                testloaders=testloaders,
                save_freq=save_frequency,
                **kwargs)

        with torch.no_grad():
            net.eval()
            for task in range(num_tasks):
                X_img = []
                Y_img_gt = []
                Y_img_hat = []
                for X, Y in testloaders[task]:
                    X = X.to(net.device)
                    Y = Y.to(net.device)
                    Y_hat = net(X, task)
                    Y_img_gt.append(Y.squeeze().cpu())
                    X_img.append((X * 27).cpu())
                    # Y_img_hat.append(Y_hat.squeeze().cpu())
                    Y_img_hat.append(torch.nn.Sigmoid()(Y_hat).squeeze().cpu())
                X_img = torch.cat(X_img)
                Y_img_gt = torch.cat(Y_img_gt)
                Y_img_hat = torch.cat(Y_img_hat)
                print(X_img.shape, Y_img_gt.shape, Y_img_hat.shape)
                imarr_gt = np.empty((28,28))
                imarr_hat = np.empty((28,28))
                imarr_gt[X_img[:,0].numpy().astype(int), X_img[:,1].numpy().astype(int)] = Y_img_gt
                imarr_hat[X_img[:,0].numpy().astype(int), X_img[:,1].numpy().astype(int)] = Y_img_hat
                print(imarr_hat.max(), imarr_hat.min(), imarr_gt.max(), imarr_gt.min())
                print(imarr_hat.sum(), imarr_gt.sum())
                plt.imshow(imarr_gt)
                plt.savefig(os.path.join(results_dir, 'tmp_img_gt_{}'.format(task)))
                plt.imshow(imarr_hat)
                plt.savefig(os.path.join(results_dir, 'tmp_img_hat_{}'.format(task)))

if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description='Qualitative experiment of MNIST reconstructions for lifelong compositional learning.')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=10, type=int)
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('-f', '--update_frequency', dest='component_update_frequency', default=100, type=int)
    parser.add_argument('--lambda', dest='ewc_lambda', default=1e-5, type=float) 
    parser.add_argument('--replay', dest='replay_size', default=-1, type=int)
    parser.add_argument('-s', '--layer_size', dest='layer_size', default=64, type=int)
    parser.add_argument('-l', '--num_layers', dest='num_layers', default=4, type=int)
    parser.add_argument('-k', '--init_tasks', dest='num_init_tasks', default=4, type=int)
    parser.add_argument('-i', '--init_mode', dest='init_mode', default='random_onehot', choices=['random_onehot', 'one_module_per_task', 'random'])
    parser.add_argument('-alg', '--algorithm', dest='algo', default='er_compositional', 
        choices=['er_compositional', 'ewc_compositional', 'van_compositional',
                'er_joint', 'ewc_joint', 'van_joint',
                'er_nocomponents', 'ewc_nocomponents', 'van_nocomponents',
                'er_dynamic', 'ewc_dynamic', 'van_dynamic',
                'fm_compositional'])
    parser.add_argument('-n', '--num_seeds', dest='num_seeds', default=1, type=int)
    parser.add_argument('-r', '--results_root', dest='results_root', default='./tmp/results')
    parser.add_argument('-sf', '--save_frequency', dest='save_frequency', default=1, type=int)
    parser.add_argument('--initial_seed', dest='initial_seed', default=0, type=int)
    parser.add_argument('--num_train', dest='num_train', default=-1, type=int)
    args = parser.parse_args()

    print('Will train on {} tasks from the {} dataset for {} epochs.'.format(args.num_tasks, 'MNISTPixels', args.num_epochs))
    print('The batch size will be {} and the modules will be updated every {} iterations'.format(args.batch_size, args.component_update_frequency))
    print('The network will contain {} layers of size {}'.format(args.num_layers, args.layer_size))
    print('The first {} tasks will be used to initialize the modules in mode {}'.format(args.num_init_tasks, args.init_mode))
    print('Experiments will be repeated for {} random seeds, starting at {}'.format(args.num_seeds, args.initial_seed))
    print('Results will be stored in {}'.format(os.path.join(args.results_root, 'MNISTPixels', args.algo)))

    main(num_tasks=args.num_tasks,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        component_update_frequency=args.component_update_frequency,
        ewc_lambda=args.ewc_lambda,
        replay_size=args.replay_size,
        layer_size=args.layer_size,
        num_layers=args.num_layers,
        num_init_tasks=args.num_init_tasks,
        init_mode=args.init_mode,
        algorithm=args.algo,
        num_seeds=args.num_seeds,
        results_root=args.results_root,
        save_frequency=args.save_frequency,
        initial_seed=args.initial_seed,
        num_train=args.num_train)