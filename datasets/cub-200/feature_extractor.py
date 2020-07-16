import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.io import loadmat, savemat
from torch.utils.data import TensorDataset, DataLoader

class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors, transform=None):
        super().__init__(tensors)
        self.transform = transform

    def __getitem__(self, idx):
        x = self.tensors[0][idx]

        if self.transform:
            x = self.transform(x)

        return x

net = models.resnet18(pretrained=True)
net.eval()
modules = list(net.children())[:-1]
net = nn.Sequential(*modules).cuda()

train = loadmat('train.mat')
X_train = train['X']
y_train = train['y']
test = loadmat('test.mat')
X_test = test['X']
y_test = test['y']
print(X_train.shape, X_test.shape)
norm_val = X_train.max()
X_train = X_train / norm_val
X_test = X_test / norm_val
print(X_train.max(), X_test.max())

X_train = torch.from_numpy(X_train).float().cuda()
X_test = torch.from_numpy(X_test).float().cuda()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trainset = CustomTensorDataset(X_train, transform=normalize)
testset = CustomTensorDataset(X_test, transform=normalize)

trainloader = DataLoader(trainset, batch_size=128, shuffle=False)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

with torch.no_grad():
    X_train_out = []
    X_test_out = []
    for X_train_mini in trainloader:
        print(X_train_mini.mean(dim=(0, 2, 3)), X_train_mini.std(dim=(0, 2, 3)))
        X_train_out.append(net(X_train_mini).squeeze().cpu())
    for X_test_mini in testloader:
        print(X_test_mini.mean(dim=(0, 2, 3)), X_test_mini.std(dim=(0, 2, 3)))
        X_test_out.append(net(X_test_mini).squeeze().cpu())

X_train_out = torch.cat(X_train_out, dim=0)
X_test_out = torch.cat(X_test_out, dim=0)
print(X_train_out.shape, X_test_out.shape)
savemat('train_resnet18.mat', {'X': X_train_out.numpy(), 'y': y_train})
savemat('test_resnet18.mat', {'X': X_test_out.numpy(), 'y': y_test})
