import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

model_path = "./Gap_net_10epoch.pth"

def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print("The number of parameters is: ", num_para)
    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


class GAPNet(nn.Module):
    """
    Insert your code here
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(6, 10, kernel_size=(5,5), stride=(1,1))
        self.global_avg_pool = nn.AvgPool2d(kernel_size=10, stride=10, padding=0)

        self.fc1 = nn.Linear(10, 10, bias=True)

    def forward(self, x):
        # print("Input:", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print("After conv1 + pool:", x.shape)
        x = self.conv2(F.relu(x))
        # print("After conv2:", x.shape)
        x = self.global_avg_pool(x)
        # print("After GAP:", x.shape)
        x = x.view(x.size(0), -1)
        # print("After flatten:", x.shape)
        x = self.fc1(x)
        # print("After FC:", x.shape)
        return x

def train_GAPNet():
    """
    Insert your code here
    """
    epochs = 10
    lr = 0.001
    momentum = 0.9
    batch_size = 4
    net = GAPNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                # print(f'[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), model_path)


def eval_GAPNet():
    """
    Insert your code here
    """
    net = GAPNet()
    net.load_state_dict(torch.load(model_path))

    batch_size=4
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

def backbone():
    """
    Insert your code here, Q3
    """
    resnet18 = models.resnet18(pretrained=True)
    # print("the structure of resnet18:\n", resnet18)
    resnet18.fc = torch.nn.Identity()

    img_path = "cat_eye.jpg"
    img = torchvision.io.read_image(img_path)
    img = img.float() / 255.0  # Normalize to [0, 1]

    transform = transforms.Compose(
        [ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    resnet18.eval()
    with torch.no_grad():
        features = resnet18(img)
    # print("Extracted features shape:", features.shape)

    return features

def transfer_learning():
    """
    Insert your code here, Q4
    """

class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    # def __init__(self, ch_in, n_classes):


    # def forward(self, x):

    
if __name__ == '__main__':
    #Q1
    resnet34 = models.resnet34(pretrained=True)
    num_para = compute_num_parameters(resnet34)
    # Q5
    ch_in=3
    n_classes=1000
    backbone()
    # model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
