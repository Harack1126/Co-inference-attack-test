import torch
from torch import nn
import torch.nn.functional as F


class InceptionA(nn.Module):
    def __init__(self, InChannels):
        super(InceptionA, self).__init__()
        self.branch1 = nn.ModuleDict(
            {
                "AvgPool": nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                "Conv_1x1": nn.Conv2d(
                    in_channels=InChannels, out_channels=24, kernel_size=1
                ),
            }
        )
        self.branch2 = nn.ModuleDict(
            {
                "Conv_1x1": nn.Conv2d(
                    in_channels=InChannels, out_channels=16, kernel_size=1
                )
            }
        )
        self.branch3 = nn.ModuleDict(
            {
                "Conv_1x1": nn.Conv2d(
                    in_channels=InChannels, out_channels=16, kernel_size=1
                ),
                "Conv_5x5": nn.Conv2d(
                    in_channels=16, out_channels=24, kernel_size=5, padding=2
                ),
            }
        )
        self.branch4 = nn.ModuleDict(
            {
                "Conv_1x1": nn.Conv2d(
                    in_channels=InChannels, out_channels=16, kernel_size=1
                ),
                "Conv_3x3_1": nn.Conv2d(
                    in_channels=16, out_channels=24, kernel_size=3, padding=1
                ),
                "Conv_3x3_2": nn.Conv2d(
                    in_channels=24, out_channels=24, kernel_size=3, padding=1
                ),
            }
        )

    def forward(self, x):
        intermediate_result = {}
        # Branch 1: AvgPool + Conv_1x1
        branch1_out = self.branch1["AvgPool"](x)
        intermediate_result["point1"] = branch1_out
        branch1_out = self.branch1["Conv_1x1"](branch1_out)

        # Branch 2: Conv_1x1
        branch2_out = self.branch2["Conv_1x1"](x)
        intermediate_result["point2"] = branch2_out

        # Branch 3: Conv_1x1 + Conv_5x5
        branch3_out = self.branch3["Conv_1x1"](x)
        branch3_out = self.branch3["Conv_5x5"](branch3_out)
        intermediate_result["point3"] = branch3_out

        # Branch 4: Conv_1x1 + 2 * Conv_3x3
        branch4_out = self.branch4["Conv_1x1"](x)
        branch4_out = self.branch4["Conv_3x3_1"](branch4_out)
        intermediate_result["point4"] = branch4_out
        branch4_out = self.branch4["Conv_3x3_2"](branch4_out)

        # Concatenate
        output = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

        return output, intermediate_result


class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        # block1 -> InceptionA1 -> block2 -> InceptionA2 -> fc
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.inceptionA1 = InceptionA(InChannels=10)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=88, out_channels=20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.inceptionA2 = InceptionA(InChannels=20)
        self.fc = nn.Linear(in_features=1408, out_features=10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x, res1 = self.inceptionA1(x)
        x = self.block2(x)
        x, res2 = self.inceptionA2(x)
        x = x.view(batch_size, -1)
        output = self.fc(x)

        return output, res1
