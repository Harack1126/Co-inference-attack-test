import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from units import get_MNIST_dataset
from model import InceptionNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def WhiteBox_Loss_Func(fx0, fx, x, beta, lmbda):
    diff_h = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2
    diff_v = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2
    diff_v = diff_v.permute(0, 1, 3, 2)
    TV_loss = (diff_h + diff_v) ** (beta / 2)
    TV_loss = TV_loss.sum()

    ED_loss = torch.norm(fx - fx0, p=2).pow(2)
    return ED_loss + lmbda * TV_loss


def get_WhiteBox_Attack_Data(data_loader, nums):
    flag = 0
    for data, _ in data_loader:
        if flag != 0:
            flag -= 1
            continue
        x0 = data[:nums]
        break
    image_size = (1, 28, 28)
    x = torch.randn(nums, *image_size)
    x = x.float().requires_grad_(True)
    return x0, x


def WhiteBox_attack_v1(model, device, dataloader, nums, epochs):
    x0, x = get_WhiteBox_Attack_Data(dataloader, nums)
    x0, x = x0.to(device), x.to(device)
    x.retain_grad()
    image_dict = {}
    for i in range(4):
        point = i + 1
        optimizer = optim.SGD([x], lr=0.01)
        for idx in range(epochs):
            _, fx = model(x)
            _, fx0 = model(x0)
            fx, fx0 = fx[f"point{point}"], fx0[f"point{point}"]
            optimizer.zero_grad()
            loss = WhiteBox_Loss_Func(fx0, fx, x, 1, 0.1)
            loss.backward()
            optimizer.step()

        np_images = x.squeeze().detach().cpu().numpy()
        image_list = [np_images[i : i + 1] for i in range(np_images.shape[0])]
        image_dict[f"point{point}"] = image_list
    return image_dict


def WhiteBox_attack_v2(model, device, dataloader, nums, epochs):
    x0, x = get_WhiteBox_Attack_Data(dataloader, nums)
    x0, x = x0.to(device), x.to(device)
    x.retain_grad()
    optimizer = optim.SGD([x], lr=0.01)
    for idx in range(epochs):
        _, fx_ = model(x)
        _, fx0_ = model(x0)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
        for i in range(4):
            point = i + 1
            fx, fx0 = fx_[f"point{point}"], fx0_[f"point{point}"]
            loss = loss + WhiteBox_Loss_Func(fx0, fx, x, 1, 0.1)
        loss.backward()
        optimizer.step()

    np_images = x.squeeze().detach().cpu().numpy()
    image_list = [np_images[i : i + 1] for i in range(np_images.shape[0])]
    return image_list


def save_WhiteBox_result(image_dict, image_nums):
    fig, axs = plt.subplots(len(image_dict), image_nums, figsize=(12, 8))
    for i, (category, images) in enumerate(image_dict.items()):
        for j, image in enumerate(images):
            image = np.squeeze(image)
            axs[i, j].imshow(image, cmap="gray")
            axs[i, j].axis("off")
            axs[i, j].set_title(f"{category}_{j}")

    plt.savefig("./result/WhiteBoxImages.png", bbox_inches="tight")
    # plt.show()
    plt.close()


def quantify_attack(original_image_list, attack_images_dict, image_nums):
    quantify_result = {}
    for idx, (category, attack_image_list) in enumerate(attack_images_dict.items()):
        psnr_list, ssim_list = [], []
        for i in range(image_nums):
            original_image = np.squeeze(original_image_list[i])
            attack_image = np.squeeze(attack_image_list[i])

            data_range = 4
            psnr = peak_signal_noise_ratio(
                original_image, attack_image, data_range=data_range
            )
            ssim = structural_similarity(
                original_image, attack_image, data_range=data_range
            )
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        quantify_result[category] = (np.mean(psnr_list), np.mean(ssim_list))
    return quantify_result


def save_quantify_result(result_dict):
    categories = list(result_dict.keys())
    values = [(psnr, ssim) for psnr, ssim in result_dict.values()]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = range(len(categories))
    ax1.set_xlabel("Point")
    ax1.set_ylabel("PSNR", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    bar1 = ax1.bar(
        index, [val[0] for val in values], bar_width, label="PSNR", color="tab:blue"
    )
    ax2 = ax1.twinx()  # 创建第二个y轴

    ax2.set_ylabel("SSIM", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    bar2 = ax2.bar(
        [i + bar_width for i in index],
        [val[1] for val in values],
        bar_width,
        label="SSIM",
        color="tab:red",
    )

    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(categories)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.savefig("./result/WhiteBoxResult.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 1000
    nums = 3
    train_loader, test_loader = get_MNIST_dataset(batch_size)
    model = InceptionNet().to(device)
    model.load_state_dict(torch.load("./models/InceptionNet.pth"))
    result_dict = WhiteBox_attack_v1(model, device, test_loader, nums, epochs)
    result_list = WhiteBox_attack_v2(model, device, test_loader, nums, epochs)
    # result_lists.append(result_list)
    result_dict["combine"] = result_list
    save_WhiteBox_result(result_dict, nums)
    original_image_list, _ = get_WhiteBox_Attack_Data(test_loader, nums)
    original_image_list = [
        original_image_list[i].numpy() for i in range(original_image_list.shape[0])
    ]
    quantify_result = quantify_attack(original_image_list, result_dict, nums)
    save_quantify_result(quantify_result)
