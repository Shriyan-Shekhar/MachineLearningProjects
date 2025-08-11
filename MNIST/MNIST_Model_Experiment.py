import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F


def prepareData():
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    return train_dataset, test_dataset

class KNNModel:
    def __init__(self, k):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

    def train(self, X_train, y_train):
        print ('Training KNN')
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def accuracy_vs_k(self, X_train, y_train, X_test, y_test):
        accuracies = []
        for k in range(1, 11):
            self.model.n_neighbors = k
            self.train(X_train, y_train)
            print (self.evaluate (X_test, y_test))
            accuracies.append(self.evaluate(X_test, y_test))
        return accuracies
    
    def plot_accuracy(self, accuracies):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), accuracies, marker='o')
        plt.title('Accuracy vs Number of Neighbors (K)')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Accuracy')
        plt.xticks(range(1, 11))
        plt.grid()
        plt.show()



class MLP(nn.Module):
    def __init__(self, hidden_neurons):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_self (self, trainLoader, epochs = 20, lr = 0.01):
        print ('Training MLP')
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in trainLoader:
                inputs = inputs.view(inputs.size(0), -1)  # This will create a shape of (64, 784)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print (f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainLoader):.4f}')

    def evaluate(self, testLoader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    
    def accuracy_vs_hidden_neurons(self, trainLoader, testLoader):
        accuracies = []
        hidden_neurons_list = [4, 8, 16, 32, 64, 128, 256]
        for hidden_neurons in hidden_neurons_list:
            self.fc1 = nn.Linear(784, hidden_neurons)
            self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
            self.fc3 = nn.Linear(hidden_neurons, 10)
            self.train_self(trainLoader)
            accuracies.append(self.evaluate(testLoader))
            print (accuracies[-1], hidden_neurons)
        return accuracies
    
    def plot_accuracy(self, accuracies):
        plt.figure(figsize=(10, 6))
        plt.plot([4, 8, 16, 32, 64, 128, 256], accuracies, marker='o')
        plt.title('Accuracy vs Number of Hidden Neurons')
        plt.xlabel('Number of Hidden Neurons')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 0, stride = 1)  # C1 number of parameters = (5*5*1+1)*6
        self.pool = nn.AvgPool2d(2, 2)  # S2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding = 0, stride = 1)  # C3 number of parameters = (5*5*6+1)*16
        self.pool2 = nn.AvgPool2d(2, 2)  # S4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten and dense layer number of parameters = (16 * 4 * 4 + 1) * 120 = 16 * 4 * 4 * 120 + 120
        self.fc2 = nn.Linear(120, 84)  # Dense layer number of parameters = 120 * 84 + 84 = 84 * (120 + 1)
        self.fc3 = nn.Linear(84, 10)  # Output layer number of parameters = (84 + 1) * 10 = 84 * 10 + 10
        self.tanh = nn.Tanh()
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm = nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.batchnorm(self.conv1(x))
        x = self.tanh(x)
        x = self.pool(x)
        x = self.batchnorm2(self.conv2(x))
        x = self.tanh(x)
        x = self.pool2(x)
        x = torch.flatten(x,1) # Flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1) 
        return x
    
    def train_self(self, train_loader, criterion, optimizer, device, epochs):
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Clear gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')

    def evaluate(self, test_loader, device):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')


class CANModel(nn.Module):
    def __init__(self):
        super (CANModel, self).__init__()   
        #size = 28 + 2* padding - (2 * dilation + 1) + 1. Output size = 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, dilation = 1, padding= 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation = 2, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation = 4, padding= 4)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation = 8, padding= 8)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, dilation = 1, padding= 1)
        self.avgpool = nn.AvgPool2d(kernel_size=28)
    
    def forward(self, x):   
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.avgpool(x)
        x = torch.flatten (x, 1)
        return x

    def train_self (self, trainLoader, epochs = 20, lr = 0.01):
        print ('Training CAN')
        criterion = nn.CrossEntropyLoss()
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0.0  
            for inputs, labels in trainLoader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainLoader):.4f}')

    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')
        

    
def main():
    train_dataset, test_dataset = prepareData()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    X_train = train_dataset.data.numpy().reshape(-1, 784)
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 784)
    y_test = test_dataset.targets.numpy()
    
    
    knn = KNNModel(k=1)
    accuracies = knn.accuracy_vs_k(X_train, y_train, X_test, y_test)
    knn.plot_accuracy(accuracies)

    print ('KNN Done')
    
    
    mlp = MLP(hidden_neurons=4)
    mlp.train_self(train_loader)
    accuracies = mlp.accuracy_vs_hidden_neurons(train_loader, test_loader)
    mlp.plot_accuracy(accuracies)

    print ('MLP Done')
    
    
    lenet = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters(), lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lenet.to(device)
    lenet.train_self(train_loader, criterion, optimizer, device, epochs=20)
    lenet.evaluate(test_loader, device)

    print ('LeNet Done')


    canModel = CANModel()
    canModel.train_self(train_loader)
    canModel.evaluate(test_loader)

    print ('CAN Done')
    
main()