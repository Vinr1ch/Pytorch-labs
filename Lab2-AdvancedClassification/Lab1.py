
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

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


def buildTFNeuralNet(x, y, eps = 6):
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    inputShape = (IS,)
    model.add(keras.layers.Dense(IS, input_shape=inputShape, activation=tf.nn.relu))  #was 2 layer, relu then sigmoid, 100 batch, 100 eps
    model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.sigmoid))
    model.compile(optimizer = opt, loss = lossType)
    if IS > 784:
        shapeOne = 50000
    else:
        shapeOne = 60000
    x = x.reshape(shapeOne, IS)
    model.fit(x, y, epochs=eps, batch_size=100)
    return model


def build10ConvNet(x, y, eps = 10, dropout = True, dropRate = 0.3):
    print("cifar_10_net")
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation = "relu"))
    model.add(keras.layers.Dense(128, activation = "relu"))
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax")) #Default is softmax`
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x, y, epochs = eps)
    return model


def build100FConvNet(x, y, eps = 20, dropout = True, dropRate = 0.3):
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation = "relu"))
    model.add(keras.layers.Dense(128, activation = "relu"))
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax")) #Default is softmax`
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x, y, epochs = eps)
    return model

def build100CConvNet(x, y, eps = 10, dropout = True, dropRate = 0.3):
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation = "relu"))
    model.add(keras.layers.Dense(128, activation = "relu"))
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax")) #Default is softmax`
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x, y, epochs = eps)
    return model

def buildMNISTConvNet(x, y, eps = 20, dropout = True, dropRate = 0.40):
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Flatten())
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.Dense(128, activation = "relu"))
    model.add(keras.layers.Dense(128, activation = "relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax")) #Default is softmax`
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x, y, epochs = eps)
    return model
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.keras.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "softmax", input_shape = inShape))
    model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = "softmax"))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(keras.layers.Flatten())
    if (dropout):
        model.add(keras.layers.Dropout(dropRate)) #https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/
    model.add(keras.layers.Dense(128, activation = "relu"))
    model.add(keras.layers.Dense(128, activation = "relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation = "softmax")) #Default is softmax`
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x, y, epochs = eps)
    return model


def buildTFConvNet(x, y, eps = 20, dropout = True, dropRate = 0.40):
    if (DATASET == 'mnist_d' or DATASET == 'mnist_f'):
        return buildMNISTConvNet(x, y)
    elif (DATASET == 'cifar_10'):
        return build10ConvNet(x, y)
    elif (DATASET == 'cifar_100_f'):
        return build100FConvNet(x, y)
    else:
        return build100CConvNet(x, y)


#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar_10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar_10.load_data()
    elif DATASET == "cifar_100_f":
        cifar_100_f = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_f.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        cifar_100_c = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_c.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
        xTrainP = xTrainP / 255
        xTestP = xTestP / 255
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
        xTrainP = xTrainP / 255
        xTestP = xTestP / 255
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
