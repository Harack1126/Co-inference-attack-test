import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LeNet
import matplotlib.pyplot as plt
from units import get_data
from attack import white_box_attack, get_test_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, optimizer, epoch):
    model.train()
    min_loss = 100
    for i in range(epoch):
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                print("Train Epoch: {} \t Loss:{:.6f}".format(i + 1, loss.item()))
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    torch.save(model.state_dict(), "./models/LeNet.pth")


def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output, _ = model(data)
        test_loss += F.cross_entropy(output, target).item()
        # test_loss += F.nll_loss(output, target, reduction="sum").item()
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


def white_box_attack_test(model, DEVICE, data_loader, num, point_list):
    save_path = "./result/"
    for image_data, target in data_loader:
        data = image_data
        break
    fig, ax = plt.subplots(1, num, figsize=(num * 2, 2))
    for i in range(num):
        image_array = data[i][0].squeeze().numpy()
        ax[i].imshow(image_array, cmap="gray")
        ax[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path + f"Original_Image")
    plt.close()

    for point in point_list:
        fig, ax = plt.subplots(1, num, figsize=(num * 2, 2))
        for i in range(num):
            recover_image = white_box_attack(model, DEVICE, data_loader, i, point)
            ax[i].imshow(recover_image, cmap="gray")
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(save_path + f"WhiteBox_Recover_Point{point}")
        plt.close()


# print(DEVICE)
if __name__ == "__main__":
    epoch = 10
    batch_size = 64
    attack_num = 4
    partition_point_list = [1, 2, 3, 4, 5, 6]
    model = LeNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    train_loader, test_loader = get_data(batch_size)
    # train_model(model, train_loader, optimizer, epoch)
    model.load_state_dict(torch.load("./models/LeNet.pth"))
    test_model(model, test_loader)
    white_box_attack_test(model, DEVICE, test_loader, attack_num, partition_point_list)
