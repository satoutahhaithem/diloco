import torch
from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR100
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch.distributed as dist


class CNNModel(nn.Module):
    def __init__(self, input_channels=1, input_height=28, input_width=28, num_classes=10):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_input_size = 64 * (input_height // 4) * (input_width // 4)

        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


class ResNetForCIFAR100(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(ResNetForCIFAR100, self).__init__()
        # Load the ResNet-18 model
        self.resnet = models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")

        # Replace the final fully connected layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)


class DiLocoWeightedAverage(DilocoSimulator):
    def _average_models(self) -> None:
        self.node_weights = [1.0] * self.config.num_nodes  # do some intelligent weighting here
        total_weight = sum(self.node_weights)

        for param in self.model.parameters():
            param.data *= self.node_weights[self.rank]
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= total_weight


# Define datasets at the module level
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
# test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

train_dataset = CIFAR100(root="./data", train=True, transform=transform, download=True)
test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=True)


if __name__ == "__main__":
    torch.manual_seed(12345)

    config = DilocoSimulatorConfig(
        model_cls=CNNModel,
        model_kwargs={"num_classes": 100, "input_channels": 3, "input_height": 32, "input_width": 32},
        optimizer_kwargs={"lr": 0.001},
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=F.cross_entropy,
        num_epochs=10,
    )

    wm = DilocoSimulator(config)
    wm.train()