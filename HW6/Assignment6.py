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
    print("The number of parameters is: ", num_para)
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
    img = img.float() / 255.0 

    transform = transforms.Compose(
        [ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)
    img = img.unsqueeze(0)  

    resnet18.eval()
    with torch.no_grad():
        features = resnet18(img)
    print("Extracted features shape:", features.shape)
    print("Extracted features:", features)

    return features

def transfer_learning():
    """
    Insert your code here, Q4
    """
    resnet18 = models.resnet18(pretrained=True)
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 10)

    for param in resnet18.parameters():
        param.requires_grad = False
    resnet18.fc.weight.requires_grad = True
    resnet18.fc.bias.requires_grad = True

    # print("structure of modified resnet18:\n", resnet18)
    
    epochs = 10
    lr = 0.001
    momentum = 0.9
    batch_size = 4

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, resnet18.parameters()), lr=lr, momentum=momentum)

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

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    torch.save(resnet18.state_dict(), "./Res_net_10epoch.pth")

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    def __init__(self, ch_in, n_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            DepthWiseConv(32, 64, stride=1),
            DepthWiseConv(64, 128, stride=2),
            DepthWiseConv(128, 128, stride=1),
            DepthWiseConv(128, 256, stride=2),
            DepthWiseConv(256, 256, stride=1),
            DepthWiseConv(256, 512, stride=2),

            DepthWiseConv(512, 512, stride=1),
            DepthWiseConv(512, 512, stride=1),
            DepthWiseConv(512, 512, stride=1),
            DepthWiseConv(512, 512, stride=1),
            DepthWiseConv(512, 512, stride=1),

            DepthWiseConv(512, 1024, stride=2),
            DepthWiseConv(1024, 1024, stride=1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.blocks(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
if __name__ == '__main__':
    #Q1
    # resnet34 = models.resnet34(pretrained=True)
    # num_para = compute_num_parameters(resnet34)
    backbone()
    # Q5
    ch_in=3
    n_classes=1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape of MobileNetV1:", y.shape)
    # transfer_learning()
    # transfer_learning()
    # model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
