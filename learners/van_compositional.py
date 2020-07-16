import torch
import torch.nn as nn
import os
from learners.base_learning_classes import CompositionalLearner

class CompositionalVAN(CompositionalLearner):
    def __init__(self, net, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)

    def update_modules(self, trainloader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)

        for X, Y in trainloader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            Y_hat = self.net(X, task_id=task_id)
            l = self.loss(Y_hat, Y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

    def update_multitask_cost(self, trainloader, task_id):
        pass