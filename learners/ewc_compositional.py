import torch
import torch.nn as nn
import os
from utils.kfac_ewc import KFAC_EWC
from learners.base_learning_classes import CompositionalLearner

class CompositionalEWC(CompositionalLearner):
    def __init__(self, net, ewc_lambda=1e-3, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        self.preconditioner = KFAC_EWC(self.net.components, ewc_lambda=ewc_lambda)

    def update_modules(self, trainloader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)

        for X, Y in trainloader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            Y_hat = self.net(X, task_id=task_id)
            l = self.loss(Y_hat, Y)
            self.optimizer.zero_grad()
            self.preconditioner.zero_grad()
            l.backward()
            self.preconditioner.step(task_id, update_stats=False, update_params=True)
            self.optimizer.step()

        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

    def update_multitask_cost(self, loader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)
        for X, Y in loader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            Y_hat = self.net(X, task_id=task_id)
            l = self.loss(Y_hat, Y)
            self.preconditioner.zero_grad()
            l.backward()
            self.preconditioner.step(task_id, update_stats=True, update_params=False)
            break

        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)
