from matplotlib import pyplot as plt
import torch
def train_LeNet(data_iter, net, optimizer, lr, loss_fn, max_epoch, batch_size, device):
    total_loss = 0
    matrix_x, matrix_loss = [], []
    batchs = len(data_iter)
    for epoch in range(max_epoch):
        for i, (X, y_target) in enumerate(data_iter):
            X, y_target = X.to(device), y_target.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y_target)
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.item()
            matrix_x.append(epoch * batchs + i + 1)
            matrix_loss.append(total_loss / (epoch * batchs + i + 1))
            print(f'loss: {matrix_loss[-1]}:{matrix_x[-1]}/{batchs * max_epoch}')

    plot_loss(matrix_x,matrix_loss)
    return net


def plot_loss(matrix_x,matrix_loss):
    plt.figure(figsize=(6, 4))
    plt.plot(matrix_x, matrix_loss, color='b', linewidth=1)
    plt.title('loss image')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()


def precise(data_iter, net, total, device):
    correct = 0
    for (X, y) in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        y_pre = torch.argmax(y_hat, dim=1)
        correct += (y_pre == y).sum().item()
    print(f"accuracy :{correct / total * 100:>7f}")