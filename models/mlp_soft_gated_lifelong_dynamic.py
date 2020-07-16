import torch
import torch.nn as nn
import numpy as np
from models.base_net_classes import SoftGatedNet

class MLPSoftGatedLLDynamic(SoftGatedNet):
    def __init__(self, 
                i_size, 
                size, 
                depth, 
                num_classes,
                num_tasks, 
                num_init_tasks=None,
                max_components=-1,
                init_ordering_mode='random',
                device='cuda',
                freeze_encoder=False,
                ):
        super().__init__(i_size,
            depth,
            num_classes,
            num_tasks,
            num_init_tasks=num_init_tasks,
            init_ordering_mode=init_ordering_mode,
            device=device)
        self.size = size
        self.max_components = max_components if max_components != -1 else np.inf
        self.num_components = self.depth
        self.freeze_encoder = freeze_encoder
        
        self.encoder = nn.ModuleList()
        for t in range(self.num_tasks):
            encoder_t = nn.Linear(self.i_size[t], self.size)
            if freeze_encoder:
                for param in encoder_t.parameters():
                    param.requires_grad = False
            self.encoder.append(encoder_t)
        
        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for i in range(self.depth):
            fc = nn.Linear(self.size, self.size)
            self.components.append(fc)

        self.decoder = nn.ModuleList()
        self.binary = False
        for t in range(self.num_tasks):
            if self.num_classes[t] == 2: self.binary = True
            decoder_t = nn.Linear(self.size, self.num_classes[t] if self.num_classes[t] != 2 else 1)
            self.decoder.append(decoder_t)

        self.structure = nn.ModuleList(nn.ModuleList(nn.Linear(self.size, self.depth) for _ in range(self.depth)) for _ in range(self.num_tasks))
        self.init_ordering()

        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)
  
    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(task_id, self.num_tasks):
                for k in range(self.depth):
                    new_node = nn.Linear(self.size, 1).to(self.device)
                    if t < task_id:
                        new_node.weight.data[:] = -np.inf
                        new_node.bias.data[:] = -np.inf
                    else:
                        assert self.structure[t][k].weight.grad is None
                        assert self.structure[t][k].bias.grad is None
                    self.structure[t][k].weight.data = torch.cat((self.structure[t][k].weight.data, new_node.weight.data), dim=0)
                    self.structure[t][k].bias.data = torch.cat((self.structure[t][k].bias.data, new_node.bias.data), dim=0)
            fc = nn.Linear(self.size, self.size).to(self.device)
            self.components.append(fc)
            self.num_components += 1

    def hide_tmp_module(self):
        self.num_components -= 1

    def recover_hidden_module(self):
        self.num_components += 1

    def remove_tmp_module(self):
        self.components = self.components[:-1]
        self.num_components = len(self.components)
        for t in range(self.num_tasks):
            for k in range(self.depth):
                self.structure[t][k].weight.data = self.structure[t][k].weight.data[:self.num_components, :]
                self.structure[t][k].bias.data = self.structure[t][k].bias.data[:self.num_components]

    def forward(self, X, task_id):
        n = X.shape[0]
        X = self.encoder[task_id](X)
        for k in range(self.depth):
            X_tmp = torch.zeros_like(X)
            s = self.structure[task_id][k](X)
            s = self.softmax(s[:, :self.num_components])   # include in the softmax only num_components (i.e., if new component is hidden, ignore it)
            for j in range(min(self.num_components, s.shape[1])):
                fc = self.components[j]
                X_tmp += s[:, j].view(-1, 1) * self.dropout(self.relu(fc(X)))

            X = X_tmp

        return self.decoder[task_id](X)  

