import  torch

def precise(test_iter, net, device):
    total = 0
    correct = 0
    for X, y in test_iter:
        total += len(X)
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        out_put = torch.argmax(y_hat, dim=-1)
        ans = (out_put == y)
        correct += ans.sum().item()

    print(correct)
    print(f"accuracy :{correct / total * 100:>3f}% ")

def train(data_iter, entroy_iter, net, optimizer, lr_scheduler, loss_fn, epochs, device, epoch_data_num):
    matrix_x, matrix_loss, entroy_loss, entroy_x = [0], [0], [], []
    total_loss = 0
    batchs = len(data_iter)
    for epoch in range(epochs):
        now_num = 0
        for X, y in data_iter:
            now_num += len(X)

            net.train()

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.sum().backward()

            optimizer.step()

            total_loss += loss.item()

            matrix_x.append(matrix_x[-1] + 1)
            matrix_loss.append(total_loss / (epoch * epoch_data_num + now_num))

            print(f"loss: {matrix_loss[-1]:>7f} now {matrix_x[-1]}/{batchs * epochs}", end='\r')

        lr_scheduler.step()
        with torch.no_grad():
            c_total_loss = 0
            test_data_num = 0
            for X, y in entroy_iter:
                net.eval()
                test_data_num += len(X)
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                loss = loss_fn(y_hat, y)

                c_total_loss += loss.item()

            print(test_data_num)
            entroy_loss.append(c_total_loss / test_data_num)
            entroy_x.append((epoch + 1) * batchs)
        print(f"cross entroy loss:{entroy_loss[-1]} now {epoch + 1}/{epochs}")

        precise(entroy_iter, net, device)

        torch.save(net.state_dict(), f"epoch{epoch + 1}.bin")

    return net, matrix_x, matrix_loss, entroy_x, entroy_loss


