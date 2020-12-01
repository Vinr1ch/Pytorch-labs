 
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
 
 
torch.manual_seed(1618)
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)
 
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"
 
#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
DATASET = "cifar_100_c"
 
if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
 
#=========================<Classifier Functions>================================
 
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)
 

class NetTFNN(nn.Module):
  def __init__(self):
    super(NetTFNN, self).__init__()
    self.lay1 = nn.Linear(IS, IS)
    self.lay2 = nn.Linear(IS, NUM_CLASSES)
  def forward(self, x):
    x = F.relu(self.lay1(x))
    x = F.sigmoid(self.lay2(x))
    return x

 
# CNN PyTorch Ciar10
import pdb
class NetCifar10C(nn.Module):
    def __init__(self):
        super(NetCifar10C, self).__init__()
        # x is 1 layer @ 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
    
    def forward(self, x):
        #pdb.set_trace()
        x = self.pool(F.relu(self.conv1(x)))  # out: 32 layers @ 16x16
        x = self.pool(F.relu(self.conv2(x)))  # out: 64 layers @ 8x8
        x = self.pool(x) # out: 64 layers @ 4x4
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.softmax(self.fc3(x))
        return x
 
def buildTorchNet(x, y, epochs = 10, dropout = True, dropRate = 0.3):
    if ALGORITHM == 'tf_net':
        net = NetTFNN()
    elif DATASET == 'cifar_10' or DATASET == 'cifar_100_c':
        net = NetCifar10C()
    elif DATASET == 'cifar_100_f':
        net = NetCifar100F()
    else:
        net = NetMnist()
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr = 0.001)
    trainLoader = torch.utils.data.DataLoader(x, batch_size=4, shuffle=True, num_workers=2)
    for epoch in range(500):
        lossTotal = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
 
            lossTotal += loss.item()
 
            if (i % 2000 == 1999):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, lossTotal / 2000))
                lossTotal = 0.0
    print("Testing TF_CNN.")
#    testLoader = torch.utils.data.DataLoader(y, batch_size=4, shuffle=False, num_workers=2)
#    total = 0
#    correct = 0
#    with torch.no_grad():
#        for data in testLoader:
#            images, labels = data
#            outputs = net(images)
#            _, predicted = torch.max(outputs.data, 1)
#            total += labels.size(0)
#            correct += (predicted == labels).sum().item()
#    print('Accuracy: %d %%' % (100 * correct / total))
    return net
 
class NetCifar100F(nn.Module):
    def __init__(self):
        super(NetCifar100F, self).__init__()
        # x is 1 layer @ 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # out: 32 layers @ 16x16
        x = self.drop(F.relu(self.conv2(x)))  # out: 64 layers @ 8x8
        x = self.pool(x) # out: 64 layers @ 4x4
        x = self.pool(x) # out: 64 layers @ 4x4
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.softmax(self.fc3(x))
        return x
 
class NetMnist(nn.Module):
    def __init__(self):
        super(NetMnist, self).__init__()
        # x is 1 layer @ 32x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
     #   pdb.set_trace()
        x = F.relu(self.conv1(x))  # out: 32 layers @ 16x16
        x = F.relu(self.conv2(x))  # out: 64 layers @ 8x8
        x = self.pool(F.relu(self.conv3(x)))  # out: 64 layers @ 8x8
        x = x.view(-1, 7744)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.softmax(self.fc3(x))
        return x
 
 
def buildTFConvNet(x, y = None, eps = 20, dropout = True, dropRate = 0.40):
      return buildTorchNet(x[0], x[1], eps, dropout, dropRate)
 
 
#=========================<Pipeline Functions>==================================
 
def getRawData():
    if DATASET == "mnist_d":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train = torchvision.datasets.MNIST(root="./data", train = True, download = True, transform = transform)
        test =  torchvision.datasets.MNIST(root="./data", train = False, download = True, transform = transform)
    elif DATASET == "mnist_f":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True, transform = transform)
        test =  torchvision.datasets.FashionMNIST(root="./data", train = False, download = True, transform = transform)
    elif DATASET == "cifar_10":
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = torchvision.datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
        test =  torchvision.datasets.CIFAR10(root="./data", train = False, download = True, transform = transform)
    elif DATASET == "cifar_100_f":
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = torchvision.datasets.CIFAR100(root="./data", train = True, download = True, transform = transform)
        test =  torchvision.datasets.CIFAR100(root="./data", train = False, download = True, transform = transform)
    elif DATASET == "cifar_100_c":
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = torchvision.datasets.CIFAR100(root="./data", train = True, download = True, transform = transform)
        test =  torchvision.datasets.CIFAR100(root="./data", train = False, download = True, transform = transform)
    else:
        raise ValueError("Dataset not recognized.")
    return (train, test)
 
 
def trainModel(data):
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFConvNet(data)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(data)
    else:
        raise ValueError("Algorithm not recognized.")
 
 
 
def runModelWithEval(data, net):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        testLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False, num_workers=2)
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        testLoader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=False, num_workers=2)
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testLoader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))
    else:
        raise ValueError("Algorithm not recognized.")
 
 
 
#=========================<Main>================================================
 
def main():
    raw = getRawData()
    data = raw
    model = trainModel(data)
    preds = runModelWithEval(data[1], model)
 
 
 
if __name__ == '__main__':
    main()