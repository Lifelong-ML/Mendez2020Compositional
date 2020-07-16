import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_net_classes import SoftGatedNet

class CNNSoftGatedLLDynamic(SoftGatedNet):
    def __init__(self, 
                i_size, 
                channels, 
                depth, 
                num_classes,
                num_tasks, 
                conv_kernel=3,
                maxpool_kernel=2,
                padding=0,
                num_init_tasks=None,
                max_components=-1,
                init_ordering_mode='random',
                device='cuda'
                ):
        super().__init__(i_size,
            depth,
            num_classes,
            num_tasks,
            num_init_tasks=num_init_tasks,
            init_ordering_mode=init_ordering_mode,
            device=device)
        self.channels = channels
        self.conv_kernel = conv_kernel
        self.padding = padding
        self.max_components = max_components if max_components != -1 else np.inf
        self.num_components = self.depth
        self.freeze_encoder = True

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(maxpool_kernel)
        self.dropout = nn.Dropout(0.5)

        out_h = self.i_size[0]
        for i in range(self.depth):
            conv = nn.Conv2d(channels, channels, conv_kernel, padding=padding)
            self.components.append(conv)
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1
        self.decoder = nn.ModuleList()
        self.binary = False
        for t in range(self.num_tasks):
            if self.num_classes[t] == 2: self.binary = True
            decoder_t = nn.Linear(out_h * out_h * channels, self.num_classes[t] if self.num_classes[t] != 2 else 1)
            self.decoder.append(decoder_t)

        self.structure = nn.ModuleList()
        self.structure_head = []
        for t in range(self.num_tasks):
            structure_t = nn.ModuleList()
            structure_head_t = []
            structure_conv = []
            for i in range(self.depth):
                conv = nn.Conv2d(channels, channels, conv_kernel, padding=padding)
                fc = nn.Linear(out_h * out_h * channels, self.depth)
                structure_conv.append(nn.Sequential(conv, self.maxpool, self.relu))
                structure_head_t.append(fc)
                structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fc))
            self.structure.append(structure_t[::-1])
            self.structure_head.append(structure_head_t[::-1])

        self.init_ordering()
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)

    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(task_id, self.num_tasks):
                for k in range(self.depth):
                    size = self.structure_head[t][k].in_features
                    new_node = nn.Linear(size, 1).to(self.device)
                    if t < task_id:
                        new_node.weight.data[:] = -np.inf
                        new_node.bias.data[:] = -np.inf
                    else:
                        assert self.structure_head[t][k].weight.grad is None
                        assert self.structure_head[t][k].bias.grad is None
                    self.structure_head[t][k].weight.data = torch.cat((self.structure_head[t][k].weight.data, new_node.weight.data), dim=0)
                    self.structure_head[t][k].bias.data = torch.cat((self.structure_head[t][k].bias.data, new_node.bias.data), dim=0)
            conv = nn.Conv2d(self.channels, self.channels, self.conv_kernel, padding=self.padding).to(self.device)
            self.components.append(conv)
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
                self.structure_head[t][k].weight.data = self.structure_head[t][k].weight.data[:self.num_components, :]
                self.structure_head[t][k].bias.data = self.structure_head[t][k].bias.data[:self.num_components]        

    def forward(self, X, task_id):
        n = X.shape[0]
        c = X.shape[1]
        X = F.pad(X, (0,0, 0,0, 0,self.channels-c, 0,0))
        for k in range(self.depth):
            X_tmp = 0.
            s = self.structure[task_id][k](X)
            s = self.softmax(s[:, :self.num_components])   # include in the softmax only num_components (i.e., if new component is hidden, ignore it)
            for j in range(min(self.num_components, s.shape[1])):
                conv = self.components[j]
                X_tmp += s[:, j].view(-1, 1, 1, 1) * self.dropout(self.relu(self.maxpool(conv(X))))
            X = X_tmp
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return self.decoder[task_id](X)         
