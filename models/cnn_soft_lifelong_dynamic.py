import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_net_classes import SoftOrderingNet

class CNNSoftLLDynamic(SoftOrderingNet):
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
                init_ordering_mode='one_module_per_task',
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

        self.to(self.device)

    def add_tmp_module(self, task_id):
        if self.num_components < self.max_components:
            for t in range(self.num_tasks):
                self.structure[t].data = torch.cat((self.structure[t].data, torch.full((1, self.depth), -np.inf if t < task_id else 1, device=self.device)), dim=0)
            conv = nn.Conv2d(self.channels, self.channels, self.conv_kernel, padding=self.padding).to(self.device)
            self.components.append(conv)
            self.num_components += 1

    def hide_tmp_module(self):
        self.num_components -= 1

    def recover_hidden_module(self):
        self.num_components += 1

    def remove_tmp_module(self):
        for s in self.structure:
            s.data = s.data[:-1, :]
        self.components = self.components[:-1]
        self.num_components = len(self.components)

    def forward(self, X, task_id):
        n = X.shape[0]
        c = X.shape[1]
        s = self.softmax(self.structure[task_id][:self.num_components, :])
        X = F.pad(X, (0,0, 0,0, 0,self.channels-c, 0,0))
        for k in range(self.depth):
            X_tmp = 0.
            for j in range(self.num_components):
                conv = self.components[j]
                X_tmp += s[j, k] * self.dropout(self.relu(self.maxpool(conv(X))))
            X = X_tmp
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return self.decoder[task_id](X)         
