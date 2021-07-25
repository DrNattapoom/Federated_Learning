# import necessary packages
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# define the device allocation
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
    class Net(nn.Module):
        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Load model
    net = Net().to(DEVICE)
    # Load data [CIFAR10 - a popular colored image classification dataset for machine learning]
    trainloader, testloader = load_data()

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        # return the model weight as a list of NumPy ndarrays
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        # update the local model weights with the parameters received from the server
        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict = True)

        # receive the updated local model weights
        def fit(self, parameters, config):
            # set the local model weights
            self.set_parameters(parameters)
            # train the local model
            train(net, trainloader, epochs = 1)
            return self.get_parameters(), len(trainloader), {}

        # test the local model
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client = CifarClient())

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    # define the loss and optimizer with PyTorch
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    net.train()
    # loop over the dataset to measure the corresponding loss and optimize it
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    # define the validation
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        # loop over the test set to measure the loss and accuracy of the test set.
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose([
        # converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) 
        transforms.ToTensor(), 
        # normalize each color channel (red, green, blue) by providing mean and std
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # using relative path may raise PermissionError: [Errno 13] Permission denied: './dataset'
    # if relative path does not work, use absolute path
    path = "./dataset"
    trainset = CIFAR10(path, train = True, download = True, transform = transform)
    testset = CIFAR10(path, train = False, download = True, transform = transform)
    trainloader = DataLoader(trainset, batch_size = 32, shuffle = True)
    testloader = DataLoader(testset, batch_size = 32)
    return trainloader, testloader

if __name__ == "__main__":
    main()
    
