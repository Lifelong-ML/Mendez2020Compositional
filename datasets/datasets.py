import struct
import pickle
from scipy.io import loadmat
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset

class SplitDataset():
    def __init__(self, num_tasks, num_classes, classes_task, with_replacement=True, flatten_images=False, normalize=True, num_train=-1):
        if not with_replacement:
            assert num_tasks <= num_classes // classes_task, 'Dataset does not support more than {} tasks'.format(num_classes // classes_task)
        self.num_tasks = num_tasks
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        if normalize:
            norm_val = X_train.max()
            X_train = X_train / norm_val
            X_test = X_test / norm_val
            X_val = X_val / norm_val

        self.trainset = []
        self.valset = []
        self.testset = []
        self.features = []

        self.max_batch_size = 0
        if not with_replacement:
            labels = np.random.permutation(num_classes)
        else:
            labels = np.array([np.random.choice(num_classes, classes_task, replace=False) for t in range(self.num_tasks)])
            labels = labels.reshape(-1)
        for task_id in range(self.num_tasks):
            
            Xb_train_t, yb_train_t, Xb_val_t, yb_val_t, Xb_test_t, yb_test_t = \
                self.split_data(X_train, y_train, X_val, y_val, X_test, y_test, labels[np.arange(task_id*classes_task, (task_id+1)*classes_task)])
            if num_train != -1:
                Xb_train_t = Xb_train_t[:num_train]
                yb_train_t = yb_train_t[:num_train]
            print(Xb_train_t.shape)
            if flatten_images:
                Xb_train_t = Xb_train_t.reshape(Xb_train_t.shape[0], -1)
                Xb_val_t = Xb_val_t.reshape(Xb_val_t.shape[0], -1)
                Xb_test_t = Xb_test_t.reshape(Xb_test_t.shape[0], -1)
            if classes_task == 2:
                yb_train_t = torch.from_numpy(yb_train_t).float().reshape(-1, 1)
                yb_val_t = torch.from_numpy(yb_val_t).float().reshape(-1, 1)
                yb_test_t = torch.from_numpy(yb_test_t).float().reshape(-1, 1)
            else:
                yb_train_t = torch.from_numpy(yb_train_t).long().squeeze()
                yb_val_t = torch.from_numpy(yb_val_t).long().squeeze()
                yb_test_t = torch.from_numpy(yb_test_t).long().squeeze()

            Xb_train_t = torch.from_numpy(Xb_train_t).float()
            Xb_val_t = torch.from_numpy(Xb_val_t).float()
            Xb_test_t = torch.from_numpy(Xb_test_t).float()
            self.trainset.append(TensorDataset(Xb_train_t, yb_train_t))
            self.valset.append(TensorDataset(Xb_val_t, yb_val_t))
            self.testset.append(TensorDataset(Xb_test_t, yb_test_t))

            if flatten_images:
                self.features.append(Xb_train_t.shape[1])
            else:
                self.features.append(Xb_train_t.shape[2])

        self.max_batch_size = 128
        self.num_classes = classes_task
    
    def split_data(self, X_train, y_train, X_val, y_val, X_test, y_test, labels):
        Xb_train = X_train[np.isin(y_train, labels)]
        yb_train = y_train.copy()
        for i in range(len(labels)):
            yb_train[y_train == labels[i]] = i
        yb_train = yb_train[np.isin(y_train, labels)]

        Xb_val = X_val[np.isin(y_val, labels)]
        yb_val = y_val.copy()
        for i in range(len(labels)):
            yb_val[y_val == labels[i]] = i
        yb_val = yb_val[np.isin(y_val, labels)]

        Xb_test = X_test[np.isin(y_test, labels)]
        yb_test = y_test.copy()
        for i in range(len(labels)):
            yb_test[y_test == labels[i]] = i
        yb_test = yb_test[np.isin(y_test, labels)]

        return Xb_train, yb_train, Xb_val, yb_val, Xb_test, yb_test

    def load_data(self):
        raise NotImplementedError('This method must be dataset specific')

class BinaryMNIST(SplitDataset):
    def __init__(self, num_tasks, num_train=-1):
        super().__init__(num_tasks, num_classes=10, classes_task=2, with_replacement=True, flatten_images=True, num_train=num_train)

    def load_data(self):
        with open('./datasets/mnist/train-labels.idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_train = np.fromfile(flbl, dtype=np.int8)

        with open('./datasets/mnist/train-images.idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_train), rows, cols)

        with open('./datasets/mnist/t10k-labels.idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_test = np.fromfile(flbl, dtype=np.int8)

        with open('./datasets/mnist/t10k-images.idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_test = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_test), rows, cols)

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]

        return X_train, y_train, X_val, y_val, X_test, y_test

class BinaryFashionMNIST(BinaryMNIST):
    '''
    Since the structure is identical to MNIST, we can simply change
    the data directory, and maintain the rest of the MNIST loaders
    '''
    def load_data(self):
        with open('./datasets/fashion/train-labels.idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_train = np.fromfile(flbl, dtype=np.int8)

        with open('./datasets/fashion/train-images.idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_train), rows, cols)

        with open('./datasets/fashion/t10k-labels.idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_test = np.fromfile(flbl, dtype=np.int8)

        with open('./datasets/fashion/t10k-images.idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_test = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_test), rows, cols)

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]

        return X_train, y_train, X_val, y_val, X_test, y_test


class SplitCIFAR100(SplitDataset):
    def __init__(self, num_tasks=20):
        super().__init__(num_tasks, num_classes=100, classes_task=5, with_replacement=False, flatten_images=False)

    def load_data(self):
        with open('./datasets/cifar-100/train', 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        y_train = np.array(data_dict[b'fine_labels'])
        X_train = data_dict[b'data'].reshape(-1, 3, 32, 32)
        s = X_train.shape

        with open('./datasets/cifar-100/test', 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        y_test = np.array(data_dict[b'fine_labels'])
        X_test = data_dict[b'data'].reshape(-1, 3, 32, 32)
        s = X_test.shape

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]
        return X_train, y_train, X_val, y_val, X_test, y_test

class SplitCUB200(SplitDataset):
    def __init__(self, num_tasks=20, resnet=True, num_train=-1):
        self.resnet = resnet
        super().__init__(num_tasks, num_classes=200, classes_task=10, with_replacement=False, flatten_images=resnet, normalize=False, num_train=num_train)

    def load_data(self):
        if self.resnet:
            train = loadmat('./datasets/cub-200/train_resnet18.mat')
        else:
            train = loadmat('./datasets/cub-200/train.mat')
        X_train = train['X']
        y_train = train['y'].squeeze()

        if self.resnet:
            test = loadmat('./datasets/cub-200/test_resnet18.mat')
        else:
            test = loadmat('./datasets/cub-200/test.mat')
        X_test = test['X']
        y_test = test['y'].squeeze()

        idx_shuffle = np.random.permutation(len(y_train))
        num_train = int(len(y_train) * .8)
        X_val = X_train[idx_shuffle[num_train:]]
        y_val = y_train[idx_shuffle[num_train:]]
        X_train = X_train[idx_shuffle[:num_train]]
        y_train = y_train[idx_shuffle[:num_train]]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

class Omniglot():
    def __init__(self, num_tasks=50, flatten_images=False):
        self.num_tasks = num_tasks

        self.trainset = []
        self.valset = []
        self.testset = []
        self.features = []
        self.num_classes = []

        self.max_batch_size = 0
        task_list = [task for task in os.listdir('./datasets/omniglot/all_tasks') if  not task.startswith('.')]
        task_order = np.random.permutation(len(task_list))
        for task_id in range(self.num_tasks):
            train = loadmat('./datasets/omniglot/all_tasks/' + task_list[task_order[task_id]] + '/train')
            val = loadmat('./datasets/omniglot/all_tasks/' + task_list[task_order[task_id]] + '/val')
            test = loadmat('./datasets/omniglot/all_tasks/' + task_list[task_order[task_id]] + '/test')

            X_train_t = train['X']
            y_train_t = train['y'].squeeze()
            print(X_train_t.shape)
            X_val_t = val['X']
            y_val_t = val['y'].squeeze()
            X_test_t = test['X']
            y_test_t = test['y'].squeeze()
            norm_val = X_train_t.max()
            X_train_t = X_train_t / norm_val
            X_val_t = X_val_t / norm_val
            X_test_t = X_test_t / norm_val

            if flatten_images:
                X_train_t = X_train_t.reshape(X_train_t.shape[0], -1)
                X_val_t = X_val_t.reshape(X_val_t.shape[0], -1)
                X_test_t = X_test_t.reshape(X_test_t.shape[0], -1)
                
            X_train_t = torch.from_numpy(X_train_t).float().unsqueeze(dim=1)
            y_train_t = torch.from_numpy(y_train_t).long()
            X_val_t = torch.from_numpy(X_val_t).float().unsqueeze(dim=1)
            y_val_t = torch.from_numpy(y_val_t).long()
            X_test_t = torch.from_numpy(X_test_t).float().unsqueeze(dim=1)
            y_test_t = torch.from_numpy(y_test_t).long()

            self.trainset.append(TensorDataset(X_train_t, y_train_t))
            self.valset.append(TensorDataset(X_val_t, y_val_t))
            self.testset.append(TensorDataset(X_test_t, y_test_t))

            self.features.append(X_train_t.shape[2])
            self.num_classes.append(int(y_train_t.max()) + 1)

        self.max_batch_size = 128


class ELLADataset():
    def __init__(self, num_tasks, num_total_tasks, path, regression=False):
        self.num_total_tasks = num_total_tasks
        self.num_tasks = min(num_tasks, num_total_tasks)

        self.trainset = []
        self.valset = []
        self.testset = []
        self.features = []
        self.num_classes = []

        self.max_batch_size = 0
        data = loadmat(path)
        task_order = np.random.permutation(self.num_total_tasks)
        X = data['feature'].squeeze()
        y = data['label'].squeeze()
        for task_id in range(self.num_tasks):
            X_t = X[task_id]
            y_t = y[task_id]
            idx_shuffle = np.random.permutation(len(y_t))
            num_train = int(len(y_t) * .5)
            num_val = int(len(y_t) * 0.)
            X_train_t = X_t[idx_shuffle[:num_train]]
            y_train_t = y_t[idx_shuffle[:num_train]]
            X_val_t = X_t[idx_shuffle[num_train:num_train+num_val]]
            y_val_t = y_t[idx_shuffle[num_train:num_train+num_val]]
            X_test_t = X_t[idx_shuffle[num_train+num_val:]]
            y_test_t = y_t[idx_shuffle[num_train+num_val:]]
            print(X_train_t.shape)

            y_train_t = torch.from_numpy(y_train_t).float()
            y_val_t = torch.from_numpy(y_val_t).float()
            y_test_t = torch.from_numpy(y_test_t).float()

            X_train_t = torch.from_numpy(X_train_t).float()
            X_val_t = torch.from_numpy(X_val_t).float()
            X_test_t = torch.from_numpy(X_test_t).float()
            self.trainset.append(TensorDataset(X_train_t, y_train_t))
            self.valset.append(TensorDataset(X_val_t, y_val_t))
            self.testset.append(TensorDataset(X_test_t, y_test_t))

            self.features.append(X_train_t.shape[1])
            self.max_batch_size = max(self.max_batch_size, X_train_t.shape[0], X_val_t.shape[0], X_test_t.shape[0])

        self.regression = regression

class Landmine(ELLADataset):
    def __init__(self, num_tasks=29):
        super().__init__(num_tasks, num_total_tasks=29, path='./datasets/ELLA/landminedata_withgroups.mat')

class FacialRecognition(ELLADataset):
    def __init__(self, num_tasks=21):
        super().__init__(num_tasks, num_total_tasks=21, path='./datasets/ELLA/fera_forella_subject_specific_pca_100_AU5_AU10_AU12.mat')

class LondonSchool(ELLADataset):
    def __init__(self, num_tasks=139):
        super().__init__(num_tasks, num_total_tasks=139, path='./datasets/ELLA/londonschools.mat')

class MNISTPixels():
    def __init__(self, num_tasks=100, digit=4):
        self.num_train = -1
        self.num_tasks = num_tasks

        self.trainset = []
        self.features = []
        self.num_classes = []

        X = self.load_data(digit)
        X = X.reshape(X.shape[0], -1)
        num_total_tasks = X.shape[0]
        self.max_batch_size = 0

        task_order = np.random.permutation(num_total_tasks)
        hw = int(np.sqrt(X.shape[1]))
        x_coord = np.tile(np.arange(hw).reshape(-1, 1), (1, hw)).reshape(-1)
        y_coord = np.tile(np.arange(hw).reshape(1, -1), (hw, 1)).reshape(-1)
        pixel_coord = np.c_[x_coord, y_coord]

        for task_id in range(self.num_tasks):
            X_train_t = pixel_coord
            y_train_t = X[task_order[task_id], :]
            print(X_train_t.shape)
            y_train_t = y_train_t / y_train_t.max()
            X_train_t = X_train_t / X_train_t.max()
            y_train_t = (y_train_t > 0.3).reshape(-1, 1)

            X_train_t = torch.from_numpy(X_train_t).float()
            y_train_t = torch.from_numpy(y_train_t).float()
            
            self.trainset.append(TensorDataset(X_train_t, y_train_t))
            self.features.append(X_train_t.shape[1])

        self.testset = self.trainset
        self.max_batch_size = 128


    def load_data(self, digit=4):
        with open('./datasets/mnist/train-labels.idx1-ubyte', 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y_train = np.fromfile(flbl, dtype=np.int8)

        with open('./datasets/mnist/train-images.idx3-ubyte', 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            X_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_train), rows, cols)
        
        idx_digit = y_train == digit
        X_train = X_train[idx_digit]
        
        return X_train
