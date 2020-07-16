import argparse 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22)
font = {'family' : 'normal',
        'size'   : 22}
import numpy as np
import os
import torch
import datasets
from mlp_soft_lifelong_pixelmnist import MLPSoftLL

SEED_SCALE = 10

def main(num_tasks,
        layer_size,
        num_layers,
        algorithms,
        results_root):
    
    if isinstance(algorithms, str):
        algorithms = [algorithms]

    dataset = 'MNISTPixels'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 0
    torch.manual_seed(seed * SEED_SCALE)
    np.random.seed(seed * SEED_SCALE)
    
    with torch.no_grad():
        torch_dataset = datasets.MNISTPixels(num_tasks)

        for algorithm in algorithms:
            learning_curves = {}
            
            ckpt_dir = os.path.join(results_root, dataset, algorithm, 'seed_{}'.format(seed), 'task_{}'.format(num_tasks - 1))
            ckpt = torch.load(os.path.join(ckpt_dir, 'checkpoint.pt'))

            net = MLPSoftLL(torch_dataset.features,
                                    size=layer_size,
                                    depth=num_layers,
                                    num_classes=torch_dataset.num_classes, 
                                    num_tasks=num_tasks,
                                    num_init_tasks=1,
                                    device=device)
            net.load_state_dict(ckpt['model_state_dict'])
            net.eval()
            for task_id, trainset in enumerate(torch_dataset.trainset):
                trainloader = (
                    torch.utils.data.DataLoader(trainset,
                        batch_size=len(trainset),
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                    ))
                X, Y = next(iter(trainloader))
                X = X.to(device)
                Y = Y.to(device)
                for layer in range(num_layers):
                    for depth in range(num_layers):
                        results_dir = os.path.join(results_root, dataset, algorithm, 'seed_{}'.format(seed), 'task_{}'.format(task_id), 'layer_{}'.format(layer), 'depth_{}'.format(depth))
                        os.makedirs(results_dir, exist_ok=True)
                        Y_hat = net.sweep(X, task_id, layer, depth)
                        for val, yhat in zip(np.linspace(0, 1, 10), Y_hat):
                            Y_img_hat = torch.nn.Sigmoid()(yhat).squeeze().cpu()
                            X_img = (X * 27).cpu()
                            imarr_hat = np.empty((28,28))
                            imarr_hat[X_img[:,0].numpy().astype(int), X_img[:,1].numpy().astype(int)] = Y_img_hat
                            plt.imshow(imarr_hat, cmap='Greys',  interpolation='nearest')
                            plt.savefig(os.path.join(results_dir, 'img_{}.png'.format(val)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create visualizations of MNIST reconstructions for lifelong compositional learning')
    parser.add_argument('-T', '--num_tasks', dest='num_tasks', default=10, type=int)
    parser.add_argument('-s', '--layer_size', dest='layer_size', default=64, type=int)
    parser.add_argument('-l', '--num_layers', dest='num_layers', default=4, type=int)
    parser.add_argument('-alg', '--algorithms', dest='algos', default='er_compositional', 
        choices=['er_compositional', 'ewc_compositional', 'van_compositional',
                'er_joint', 'ewc_joint', 'van_joint',
                'er_nocomponents', 'ewc_nocomponents', 'van_nocomponents',
                'er_dynamic', 'ewc_dynamic', 'van_dynamic',
                'fm_compositional', 'fm_dynamic'],
                nargs='+')
    parser.add_argument('-r', '--results_root', dest='results_root', default='./tmp/results')
    args = parser.parse_args()

    main(args.num_tasks,
        args.layer_size,
        args.num_layers,
        args.algos,
        args.results_root)