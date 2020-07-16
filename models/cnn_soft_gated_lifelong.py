import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_net_classes import SoftGatedNet

class CNNSoftGatedLL(SoftGatedNet):
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
        for t in range(self.num_tasks):
            structure_t = nn.ModuleList()
            structure_conv = []
            for i in range(self.depth):
                conv = nn.Conv2d(channels, channels, conv_kernel, padding=padding)
                structure_conv.append(nn.Sequential(conv, self.maxpool, self.relu))
                structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), nn.Linear(out_h * out_h * channels, self.depth)))
            self.structure.append(structure_t[::-1])

        self.init_ordering()
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)

    def forward(self, X, task_id):
        n = X.shape[0]
        c = X.shape[1]
        X = F.pad(X, (0,0, 0,0, 0,self.channels-c, 0,0))
        for k in range(self.depth):
            X_tmp = 0.
            s = self.softmax(self.structure[task_id][k](X))
            for j in range(self.depth):
                conv = self.components[j]
                X_tmp += s[:, j].view(-1, 1, 1, 1) * self.dropout(self.relu(self.maxpool(conv(X))))
            X = X_tmp
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return self.decoder[task_id](X)         
