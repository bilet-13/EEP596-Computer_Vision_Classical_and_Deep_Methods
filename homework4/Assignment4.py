import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    images, labels = next(iter(trainloader))
    imshow(torchvision.utils.make_grid(images), labels, classes)
    
    return images, labels

def imshow(img, labels, classes):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    for i in range(len(labels)):
        plt.text(i * 32 + 16, 0, classes[labels[i].item()],
                 color='black', ha='center', va='bottom', fontsize=8, weight='bold')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# show images
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_classifier():
    # Creates dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # Creates Network 
    net = Net()

    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset for 2 iterations
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    # Saves the model weights after training
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # Loads dataset
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
    # since we're not training, we 
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    return 100.0 * correct / total

def get_first_layer_weights():
    PATH = './cifar_net_2epoch.pth'

    net = Net()
    net.load_state_dict(torch.load(PATH))
    # TODO: load the trained weights
    first_weight = net.conv1.weight  # TODO: get conv1 weights (exclude bias)
    return first_weight

def get_second_layer_weights():
    PATH = './cifar_net_2epoch.pth'

    net = Net()
    net.load_state_dict(torch.load(PATH))
    # TODO: load the trained weights
    second_weight = net.conv2.weight  # TODO: get conv2 weights (exclude bias)
    return second_weight

def get_error(model, dataset, batch_size, num_workers, n):
    """Evaluate model on a random subset of size n (no replacement). Returns (avg_loss, error_pct)."""
    model.eval()
    n = min(n, len(dataset))
    sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=n)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images
            labels = labels
            outputs = model(images)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    error_pct = 100.0 * (1.0 - correct / total)
    return  error_pct

def hyperparameter_sweep():
    """
    Train with learning rates: 0.01, 0.001, 0.0001
    Every 2000 minibatches: record training loss, and compute train/test errors by sampling 1000 images from each dataset.
    Plot curves at the end.
    """
    # Dataset / loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    batch_size = 4
    num_workers = 2

    trainset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=False, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root="./cifar10", train=False, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)

    learning_rates = [0.01, 0.001, 0.0001]
    training_losses = {lr: [] for lr in learning_rates}  
    train_errors    = {lr: [] for lr in learning_rates} 
    test_errors     = {lr: [] for lr in learning_rates}

    # ----- sweep -----
    for lr in learning_rates:
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        net.train()
        for epoch in range(2):  
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs = inputs
                labels = labels

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # every 2000 mini-batches
                if i % 2000 == 1999:
                    avg_train_loss_block = running_loss / 2000.0
                    training_losses[lr].append(avg_train_loss_block)

                    print(f"LR={lr} [epoch {epoch+1}, iter {i+1}] block train loss: {avg_train_loss_block:.4f}")
                    running_loss = 0.0

                    train_err_1k = get_error(
                        net, trainset, batch_size, num_workers, n=1000)
                    test_err_1k  = get_error(
                        net, testset,  batch_size, num_workers, n=1000  )

                    train_errors[lr].append(train_err_1k)
                    test_errors[lr].append(test_err_1k)

                    # keep training mode for next iterations
                    net.train()
    draw_picture(learning_rates, training_losses, 
                 title="Training Loss",
                 xlabel="Iterations (per 2000 mini-batches)",
                 ylabel="Training Loss",
                 picture_name="training_loss.png")
    draw_picture(learning_rates, train_errors, 
                 title="Training Error",
                 xlabel="Iterations (per 2000 mini-batches)",
                 ylabel="Training Error (%)",
                 picture_name="train_error.png")
    draw_picture(learning_rates, test_errors, 
                 title="Test Error ",
                 xlabel="Iterations (per 2000 mini-batches)",
                 ylabel="Test Error (%)",
                 picture_name="test_error.png")
    return None


def draw_picture(learning_rates, points,  title, xlabel, ylabel, picture_name):
    plt.figure()
    for lr in learning_rates:
        plt.plot(points[lr], label=f"LR={lr}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(picture_name, dpi=150)
    plt.show()

    return None
if __name__ == "__main__":
    # weight1 = get_first_layer_weights()
    # weight2 = get_second_layer_weights()
    # images, labels = CIFAR10_dataset_a()
    hyperparameter_sweep()
