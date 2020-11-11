import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"



#(784, 10, 512);

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.008):
        self.inputSize = inputSize  #784
        self.outputSize = outputSize  #10
        self.neuronsPerLayer = neuronsPerLayer  #512
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):

        return 1 / (1 + np.exp(-x))   #TODO: implement should be done

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x * (1 - x)  #TODO: implement should be done

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 200, minibatches = True, mbs = 100):
        #training the model to make accurate predictions while adjusting weights continually

        for iteration in range(epochs):
            gen1 = self.__batchGenerator(xVals, mbs)
            gen2 = self.__batchGenerator(yVals, mbs)
            for iteration in range(600):
                batchx = next(gen1)

                batchy = next(gen2)

                output = self.predict(batchx)


                #computing error rate for back-propagation
                error = batchy - output

                delta = error * self.__sigmoidDerivative(output)
                 # applying derivative of sigmoid to error
                z2_error = delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
                z2_delta = z2_error*self.__sigmoidDerivative(self.layer1) # derivative of sigmoid
                #performing weight adjustments
                tempW = np.copy(self.W1)
                self.W1 += batchx.T.dot(z2_delta) * self.lr # adjusting first weights

                self.W2 += self.layer1.T.dot(delta) * self.lr# adjusting second weights
                #comparsion = self.W1 == tempW
                #if comparsion.all():
                #    print(True)
                #else:
                #    print(False)
                                              #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.




    # Forward pass.
    def __forward(self, input):
        input = input.astype(float)

        self.layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(self.layer1, self.W2))
        return self.layer1, layer2

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



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    xTrainP = xTrain.reshape((60000, 784))
    xTestP = xTest.reshape((10000, 784))
    xTrain, xTest = xTrain / 255.0, xTest / 255.0
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    NeuralNetwork = NeuralNetwork_2Layer(784, 10, 512);
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("implemented.")                   #TODO: Write code to build and train your custon neural net.
        NeuralNetwork.train(xTrain, yTrain)
        print("Ending Weights After Training: ")
        print(NeuralNetwork.W1)
        print(NeuralNetwork.W2)
        preds = NeuralNetwork.predict(xTrain)

        yhat=np.argmax(preds, axis=1)
        print(yhat)
        yTestNew = np.argmax(yTrain, axis=1)
        print(yTestNew)
        acc = 0
        # accuracy: (tp + tn) / (p + n)
        accuracy = int(sum(yTestNew == yhat) / len(yTestNew) * 100)
        print('Accuracy: %f' % accuracy)

        return NeuralNetwork
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(str(xTrain.shape))
        print(str(yTrain.shape))


        model.fit(xTrain, yTrain, epochs=10)
        print(model.evaluate(xTrain, yTrain))
                         #TODO: Write code to build and train your keras neural net.
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    xTest = data
    #NeuralNetwork = NeuralNetwork_2Layer(784, 10, 512);
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        #NeuralNetwork.predict(data)
        #print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("# DEBUG: ")
        print(model.predict_classes(xTest))
        return model.predict_classes(xTest)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    if ALGORITHM == "guesser":
        xTest, yTest = data
        acc = 0
        for i in range(preds.shape[0]):
            if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        accuracy = acc / preds.shape[0]
        print("Classifier algorithm: %s" % ALGORITHM)
        print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()
    elif ALGORITHM == "custom_net":
        xTest, yTest = data
        yhat=np.argmax(preds, axis=1)
        yTestNew = np.argmax(yTest, axis=1)
        acc = 0
        # accuracy: (tp + tn) / (p + n)
        accuracy = int(sum(yTestNew == yhat) / len(yTestNew) * 100)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(yTestNew, yhat, pos_label = 'positive', average='micro')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(yTestNew, yhat, pos_label='positive', average='micro')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(yTestNew, yhat, pos_label='positive', average='micro')
        print('F1 score: %f' % f1)
        cm = confusion_matrix(yTestNew, yhat)
        print(cm)

        #for i in range(preds.shape[0]):
        #    if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        #accuracy = acc / preds.shape[0]
        print("Classifier algorithm: %s" % ALGORITHM)
        #print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()                #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        xTest, yTest = data
        yhat=preds
        yTestNew = np.argmax(yTest, axis=1)
        acc = 0
        # accuracy: (tp + tn) / (p + n)
        accuracy = int(sum(yTestNew == yhat) / len(yTestNew) * 100)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(yTestNew, yhat, pos_label = 'positive', average='micro')
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(yTestNew, yhat, pos_label='positive', average='micro')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(yTestNew, yhat, pos_label='positive', average='micro')
        print('F1 score: %f' % f1)
        cm = confusion_matrix(yTestNew, yhat)
        print(cm)

        #for i in range(preds.shape[0]):
        #    if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        #accuracy = acc / preds.shape[0]
        print("Classifier algorithm: %s" % ALGORITHM)
        #print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()


        return None
    else:
        raise ValueError("Algorithm not recognized.")



#=========================<Main>================================================

def main():

    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
