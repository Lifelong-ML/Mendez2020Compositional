import torch
import torch.nn as nn
import os
from learners.base_learning_classes import NoComponentsLearner

class NoComponentsVAN(NoComponentsLearner):
    def __init__(self, net, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=eval_bool)
        if self.T <= self.net.num_init_tasks:
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            iter_cnt = 0
            for i in range(num_epochs):
                for X, Y in trainloader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    self.gradient_step(X, Y, task_id)
                    iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)

    def update_multitask_cost(self, loader, task_id):
        pass