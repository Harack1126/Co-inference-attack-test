import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


def get_test_image(data_loader, index):
    for data, target in data_loader:
        x0 = data[index]
        break
    x0 = x0.unsqueeze(1)
    image_size = (28, 28)
    x = np.random.rand(*image_size)
    x = torch.tensor(x, requires_grad=True)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x = x.float()
    # print(x.size(), x0.size())
    return x0, x


def white_box_attack(model, device, data_loader, index, point):
    # x0: target, x: generate
    x0, x = get_test_image(data_loader, index)
    x0, x = x0.to(device), x.to(device)
    _, fx0 = model(x0)
    fx0 = fx0["point1"][0]
    x.retain_grad()
    optimizer = optim.SGD([x], lr=0.01)

    epochs = 1000

    for epoch in range(epochs):
        _, fx = model(x)
        fx = fx[f"point{point}"][0]
        _, fx0 = model(x0)
        fx0 = fx0[f"point{point}"][0]
        optimizer.zero_grad()
        loss = torch.norm(fx - fx0, p=2) ** 2
        loss.backward()
        optimizer.step()

    np_image = x.squeeze().detach().cpu().numpy()
    return np_image
