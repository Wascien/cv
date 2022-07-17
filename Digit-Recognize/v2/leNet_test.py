import torch
import torch.nn as nn
import torch.nn.functional as F
from DataModel import MinistData
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10))

    def forward(self, X):
        return self.net(X)


Test_data = MinistData(
    'D:/PY/hand_written/data/MNIST/raw/train-images-idx3-ubyte',
    'D:/PY/hand_written/data/MNIST/raw/train-labels-idx1-ubyte')


# def train(data_iter, net, optimizer, lr, loss_fn, max_epoch, batch_size, device):
#     total_loss = 0
#     matrix_x, matrix_loss = [], []
#     batches = len(data_iter)
#     for epoch in range(max_epoch):
#         for i, (X, y_target) in enumerate(data_iter):
#             X, y_target = X.to(device), y_target.to(device)
#             optimizer.zero_grad()
#             y_hat = net(X)
#             loss = loss_fn(y_hat, y_target)
#             loss.sum().backward()
#             optimizer.step()
#             total_loss += loss.item()
#             matrix_x.append(epoch * batches + i + 1)
#             matrix_loss.append(total_loss / (epoch * batches + i + 1))
#             print(f'loss: {matrix_loss[-1]}:{matrix_x[-1]}/{batches * max_epoch}')
#
#     torch.save(net.state_dict(), './model.pth')
#     torch.save(optimizer.state_dict(), './optimizer.pth')
#     return net

def strain(data_iter, net, optimizer, loss_fn, lr, batch_size, device, max_epoch,scheduler):
    for epoch in range(max_epoch):
        for i, (X, Y) in enumerate(data_iter):
            scheduler.step()
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            Y_out = net(X)
            loss = loss_fn(Y_out, Y)
            loss.sum().backward()
            optimizer.step()
    torch.save(net.state_dict(),'./model.bin')
    torch.save(optimizer.state_dict(),'./optimizer.bin')
    return net


device = "cuda" if torch.cuda.is_available() else "cpu"
net = net().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=3e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5500, gamma=0.5, last_epoch=-1)
data_iter = DataLoader(Test_data, batch_size=64, shuffle=True)
strain(data_iter, net, optimizer, lr=3e-2, loss_fn=loss_fn, max_epoch=10, batch_size=64, device=device,scheduler = scheduler)


test_data = MinistData('D:/PY/hand_written/data/MNIST/raw/t10k-images-idx3-ubyte',
                     'D:/PY/hand_written/data/MNIST/raw/t10k-labels-idx1-ubyte')


def prediction(data_iter, total, device):
    correct = 00
    for(X,y) in data_iter:
        X,y = X.to(device),y.to(device)
        y_hat = net(X)
        y_pre = torch.argmax(y_hat, dim=1)
        correct+=(y_pre==y).sum().item()
    print((f"accuracy:{correct/total*100}"))

data_iter = DataLoader(test_data, batch_size=64,shuffle=True)
prediction(data_iter, len(test_data),device=device)










