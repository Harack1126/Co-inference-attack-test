import torch
import torch.optim as optim
from model import InceptionNet
import torch.nn.functional as F
from units import get_MNIST_dataset


def train(model, device, optimizer, train_loader, epochs):
    model.train()
    min_loss = 100
    for i in range(epochs):
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print("Train Epoch: {} \t Loss:{:.6f}".format(i + 1, loss.item()))
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(model.state_dict(), "./models/InceptionNet.pth")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, _ = model(data)
        test_loss += F.cross_entropy(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    batch_size = 64
    attack_num = 4
    model = InceptionNet().to(device)
    optimizer = optim.Adam(model.parameters())
    train_loader, test_loader = get_MNIST_dataset(batch_size)
    train(model, device, optimizer, train_loader, epochs)
    model.load_state_dict(torch.load("./models/InceptionNet.pth"))
    test(model, device, test_loader)
