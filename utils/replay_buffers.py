import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

class ReplayBufferBase(TensorDataset):
    def __init__(self, memory_size):
        self.observed = 0
        self.memory_size = memory_size
        self.is_dataset_init = False

    def push(self, *tensors):
        raise NotImplementedError('Implementation must be specific to each buffer type')

    def __len__(self):
        return min(self.observed, self.memory_size)

class ReplayBufferReservoir(ReplayBufferBase):
    def __init__(self, memory_size, task_id):
        super().__init__(memory_size)
        self.task_id = task_id

    def push(self, X, Y):
        if not self.is_dataset_init:
            self.tensors = (torch.empty(self.memory_size, *X.shape[1:], device=X.device, dtype=X.dtype), 
                torch.empty(self.memory_size, *Y.shape[1:], device=Y.device, dtype=Y.dtype),
                torch.full((self.memory_size,), self.task_id, dtype=int))
            self.is_dataset_init = True
        for j, (x, y) in enumerate(zip(X, Y)):
            if self.observed + j < self.memory_size:
                self.tensors[0].data[self.observed + j] = x
                self.tensors[1].data[self.observed + j] = y
            else:
                idx = np.random.randint(self.observed + j)
                if idx < self.memory_size:
                    self.tensors[0].data[idx] = x
                    self.tensors[1].data[idx] = y
            j += 1
        self.observed += j

class ReplayBufferRing(ReplayBufferBase):
    def __init__(self):
        super().__init__(memory_size)
        self.last_idx = 0

    def push(self, X, Y):
        if not self.is_dataset_init:
            self.tensors = (torch.empty(self.memory_size, *X.shape[1:], torch.empty(self.memory_size, *Y.shape[1:])))
            self.is_dataset_init = True
        for x, y in zip(X, Y):
            self.tensors[0].data[self.last_idx] = x
            self.last_idx = (self.last_idx + 1) % self.memory_size
            self.observed += 1
