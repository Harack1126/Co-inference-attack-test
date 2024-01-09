from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.level1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.level2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
        )
        self.level4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )
        self.level5 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.level6 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(84, 10),
        )

    def forward(self, x):
        intermediate_result = {}
        x = self.level1(x)
        intermediate_result["point1"] = x
        x = self.level2(x)
        intermediate_result["point2"] = x
        x = self.level3(x)
        intermediate_result["point3"] = x
        x = self.level4(x)
        intermediate_result["point4"] = x
        x = x.view(x.size(0), -1)
        x = self.level5(x)
        intermediate_result["point5"] = x
        x = self.level6(x)
        intermediate_result["point6"] = x
        x = self.classifier(x)
        return x, intermediate_result
