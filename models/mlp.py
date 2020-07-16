import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
                i_size, 
                size, 
                depth, 
                num_classes,
                num_tasks, 
                num_init_tasks,
                device='cuda',
                freeze_encoder=False,
                ):
        super().__init__()
        self.device = device
        self.size = size
        self.depth = depth
        self.num_tasks = num_tasks
        self.num_init_tasks = num_init_tasks
        if isinstance(num_classes, int):
            num_classes = [num_classes] * self.num_tasks
        self.freeze_encoder = freeze_encoder
        
        if isinstance(i_size, int):
            i_size = [i_size] * num_tasks
        self.encoder = nn.ModuleList()
        for t in range(self.num_tasks):
            encoder_t = nn.Linear(i_size[t], self.size)
            for param in encoder_t.parameters():
                param.requires_grad = not freeze_encoder
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
            if num_classes[t] == 2: self.binary = True
            decoder_t = nn.Linear(self.size, num_classes[t] if num_classes[t] != 2 else 1)
            self.decoder.append(decoder_t)

        self.to(self.device)

    def forward(self, X, task_id):
        n = X.shape[0]
        X = self.encoder[task_id](X)
        for fc in self.components:
            X = self.dropout(self.relu(fc(X)))

        return self.decoder[task_id](X)        
