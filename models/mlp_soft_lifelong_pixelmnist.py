import torch
import torch.nn as nn
import numpy as np

class MLPSoftLL(nn.Module):
    def __init__(self, 
                i_size, 
                size, 
                depth, 
                num_classes,
                num_tasks, 
                num_init_tasks=None,
                init_ordering_mode='one_module_per_task',
                device='cuda'
                ):
        super().__init__()
        self.device = device
        self.depth = depth
        self.num_tasks = num_tasks
        if num_init_tasks is None:
            num_init_tasks = depth
        self.num_init_tasks = num_init_tasks
        self.init_ordering_mode = init_ordering_mode
        self.i_size = i_size
        if isinstance(self.i_size, int):
            self.i_size = [self.i_size] * num_tasks
        self.num_classes = num_classes
        if isinstance(self.num_classes, int):
            self.num_classes = [self.num_classes] * num_tasks

        self.size = size
        self.freeze_encoder = True
        
        self.encoder = nn.Linear(self.i_size[0], self.size)
        
        self.structure = nn.ParameterList([nn.Parameter(torch.ones(self.depth, self.depth)) for t in range(self.num_tasks)])

        self.init_ordering()

        self.softmax = nn.Softmax(dim=0)
        self.components = nn.ModuleList()
        self.relu = nn.ReLU()

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.Linear(self.size, 1)
        
        self.regression = False
        self.binary = not self.regression
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)


        ###### Hacky way of handling freeze structure so that it doesn't freeze during initialization
        self.freeze_structure_cnt = 0

    def init_ordering(self):
        pass

    def freeze_modules(self, freeze=True):
        for param in self.components.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

        for param in self.decoder.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None

    def freeze_structure(self, freeze=True, task_id=None):
        '''
        Since we are using Adam optimizer, it is important to
        set requires_grad = False for every parameter that is 
        not currently being optimized. Otherwise, even if they
        are untouched by computations, their gradient is all-
        zeros and not None, and Adam counts it as an update.
        '''
        ###### Hacky way of handling freeze structure so that it doesn't freeze during initialization
        if self.freeze_structure_cnt < self.num_init_tasks:
            self.freeze_structure_cnt += 1
            print('Not freezing structure...')
            return
        print('Freezing/unfreezing structure...')
        if task_id is None:
            for param in self.structure:
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
        else:
            self.structure[task_id].requires_grad = not freeze
            if freeze:
                self.structure[task_id].grad = None

    def sweep(self, X, task_id, layer=0, depth=0):
        n = X.shape[0]
        s = self.softmax(self.structure[task_id])
        predictions = []
        X_copy = X.data.clone()
        for val in np.linspace(0, 1, 10):
            X = self.encoder(X_copy)
            s[layer, depth] = val
            for k in range(self.depth):
                X_tmp = torch.zeros_like(X)
                for j in range(self.depth):
                    fc = self.components[j]
                    X_tmp += s[j, k] * self.relu(fc(X))
                X = X_tmp

            predictions.append(self.decoder(X))
        return predictions


    def forward(self, X, task_id):
        n = X.shape[0]
        s = self.softmax(self.structure[task_id])
        X = self.encoder(X)
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            for j in range(self.depth):
                fc = self.components[j]
                X_tmp += s[j, k] * self.relu(fc(X))
            X = X_tmp
        return self.decoder(X)


