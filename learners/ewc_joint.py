import torch
import torch.nn as nn
import os
from utils.kfac_ewc import KFAC_EWC
from learners.base_learning_classes import JointLearner

class JointEWC(JointLearner):
    def __init__(self, net, ewc_lambda=1e-3, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        self.preconditioner = KFAC_EWC(self.net.components, ewc_lambda=ewc_lambda)

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=eval_bool)
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.freeze_structure(freeze=False, task_id=task_id)    # except current one
            iter_cnt = 0
            for i in range(num_epochs):
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
                    iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            self.update_multitask_cost(trainloader, task_id)

    def update_multitask_cost(self, loader, task_id):
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
        self.net.freeze_structure(freeze=False, task_id=task_id)

