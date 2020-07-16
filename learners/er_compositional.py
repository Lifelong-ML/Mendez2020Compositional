import torch
import torch.nn as nn
import os
from torch.utils.data.dataset import ConcatDataset
import copy
from utils.replay_buffers import ReplayBufferReservoir
from learners.base_learning_classes import CompositionalLearner


class CompositionalER(CompositionalLearner):
    def __init__(self, net, memory_size, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        self.replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def update_modules(self, trainloader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)
        prev_reduction = self.loss.reduction 
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances

        tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset([loader.dataset for loader in self.memory_loaders.values()] + [tmp_dataset])
        tmp_loader = next(iter(self.memory_loaders.values()))
        batch_size = trainloader.batch_size
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
            )
        for X, Y, t in mega_loader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            l = 0.
            n = 0
            all_t = torch.unique(t)
            for task_id_tmp in all_t:
                Y_hat = self.net(X[t == task_id_tmp], task_id=task_id_tmp)
                l += self.loss(Y_hat, Y[t == task_id_tmp])
                n += X.shape[0]
            l /= n
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

        self.loss.reduction = prev_reduction
        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)    # unfreeze only current task's structure

    def update_multitask_cost(self, trainloader, task_id):
        self.replay_buffers[task_id] = ReplayBufferReservoir(self.memory_size, task_id)
        for X, Y in trainloader:
            self.replay_buffers[task_id].push(X, Y)
        self.memory_loaders[task_id] =  (
            torch.utils.data.DataLoader(self.replay_buffers[task_id],
                batch_size=trainloader.batch_size,
                shuffle=True,
                num_workers=10,
                pin_memory=True
                ))
