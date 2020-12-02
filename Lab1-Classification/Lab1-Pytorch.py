from __future__ import print_function
import os
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import *
from matplotlib import pyplot as plt
#resources used was pytroch api
#https://pytorch.org/docs/stable/index.html

# Setting random seeds to keep everything deterministic.
#removed as it was unnneeded
#random.seed(1618)
#np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
#tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "pytorch_NN"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        pass   #TODO: implement

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        pass   #TODO: implement

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#pytorch Functions


def train_pytroch(model, device, train_loader, opt, epoch):
    loss = []
    model.train()
    #loop used for training
    for batch_idx, (data, target) in enumerate(train_loader):
        #load data and target and change to device being used
        data, target = data.to(device), target.to(device)
        #update optimizer with zero_grads
        opt.zero_grad()
        #run model with data
        output = model(data)
        #calulate losses
        loss = F.nll_loss(output, target)
        loss.backward()
        opt.step()
        #updates loss and append information
        loss.append(loss.item())
        #update imformation every 100 batches
        if batch_idx > 0 and batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}\t({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss


def test_pytorch(model, device, test_loader):
    #evalute the model
    model.eval()
    #variable to hold number of correct and total total loss of program
    t_l = 0
    n_corr = 0

    with torch.no_grad():
        #loop to run model on test set
        for data, target in test_loader:
            #load data in varibales
            data, target = data.to(device), target.to(device)
            #get output from model
            output = model(data)
            # get total sum of batch loss
            t_l += F.nll_loss(output, target, reduction='sum').item()
            # get the index of log-probability
            pred = output.argmax(dim=1, keepdim=True)
            n_corr += pred.eq(target.view_as(pred)).sum().item()
    t_l /= len(test_loader.dataset)
    #output data to printed about average loss and accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        t_l, n_corr, len(test_loader.dataset),
        100. * n_corr / len(test_loader.dataset)))
    #return data to be used in output for evaluation
    return (float(n_corr) / len(test_loader.dataset))

def test_label_predictions(model, device, test_loader):
    model.eval()
    #create arrays to hold values for confusion_matrix
    actuals = []
    predictions = []

    with torch.no_grad():
        #loop used to update data and get accuracy of model
        for data, target in test_loader:
            #load data in varibales
            data, target = data.to(device), target.to(device)
            #get output from model
            output = model(data)
            # get the index of log-probability
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

#=========================<Pipeline Functions>==================================

def getRawData():
    #load datat from MNIST from pytorch
    #Training set to true, also normalize data for use
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
           train=True,
           download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])
        ),
        batch_size=64,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=1000,
        shuffle=True)
    return (train_loader, test_loader)


"""
#removec as it is unneeded with program, should be changed for custom NN
def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

"""



def trainModel(data):
    train_loader = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return None
    elif ALGORITHM == "pytorch_NN":
        model = CNN()
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        device = torch.device("cpu") # or 'gpu' can be changed as needed

        #array used for outpur used later to analyze model
        loss = []
        we_right = []
        for epoch in range(0, 5):
            loss.extend(train_pytroch(model, device, train_loader, opt, epoch))
            we_right.append(test_pytorch(model, device, train_loader))
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "pytorch_NN":
        test_loader = data
        print("Testing TF_NN.")
        device = torch.device("cpu") # or 'gpu' can be changed as needed
        actuals, predictions = test_label_predictions(model, device, test_loader)
        return actuals, predictions
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, actuals, predictions):
    #this is use for confusion_matrix output, but will not workon custom NN
    if ALGORITHM == "pytorch_NN":
        print('Confusion matrix:')
        print(confusion_matrix(actuals, predictions))
        print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
        print('Accuracy score: %f' % accuracy_score(actuals, predictions))



#=========================<Main>================================================

def main():
    data = getRawData()
    #removed as it was unneeded from
    #data = preprocessData(raw)
    model = trainModel(data[0])
    actuals, predictions = runModel(data[1], model)
    evalResults(data, actuals, predictions)



if __name__ == '__main__':
    main()
