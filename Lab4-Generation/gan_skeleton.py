# CS390-NIP GAN lab
# Max Jacobson / Sri Cherukuri / Anthony Niemiec
# FA2020
# uses Fashion MNIST https://www.kaggle.com/zalando-research/fashionmnist 
# uses CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import Adam
#from scipy.misc import imsave
import random
from PIL import Image
import matplotlib.pyplot as plt

random.seed(1618)
np.random.seed(1618)
tf.compat.v1.set_random_seed(1618)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# NOTE: mnist_d is no credit
# NOTE: cifar_10 is extra credit
#DATASET = "mnist_d"
#DATASET = "mnist_f"
DATASET = "cifar_10"

if DATASET == "mnist_d":
    IMAGE_SHAPE = (IH, IW, IZ) = (28, 28, 1)
    LABEL = "numbers"

elif DATASET == "mnist_f":
    IMAGE_SHAPE = (IH, IW, IZ) = (28, 28, 1)
    CLASSLIST = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    # TODO: choose a label to train on from the CLASSLIST above
    LABEL = "ankle boot"

elif DATASET == "cifar_10":
    IMAGE_SHAPE = (IH, IW, IZ) = (32, 32, 3)
    CLASSLIST = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    LABEL = "deer"

IMAGE_SIZE = IH*IW*IZ

NOISE_SIZE = 100    # length of noise array

# Ratio ex: if 1:2 ration of discriminator:generator, set adv_ratio = 2 and gen_ratio = 1
# Implementation uses mod to determine if somthing gets trained. i.e. if adv_ratio is set to 2, it will train every other epoch
USE_RATIO = 0
adv_ratio = 2
gen_ratio = 1

alpha_relu = 0.1

gen_losses_plot = [[], []]
adv_losses_plot = [[], []]
#epochs_to_view_plot = 5000

# file prefixes and directory
OUTPUT_NAME = DATASET + "_" + LABEL
OUTPUT_DIR = "./outputs/" + OUTPUT_NAME

# NOTE: switch to True in order to receive debug information
VERBOSE_OUTPUT = False

################################### DATA FUNCTIONS ###################################

# Load in and report the shape of dataset
def getRawData():
    if DATASET == "mnist_f":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
    elif DATASET == "cifar_10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "mnist_d":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

# Filter out the dataset to only include images with our LABEL, meaning we may also discard
# class labels for the images because we know exactly what to expect
def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if DATASET == "mnist_d":
        xP = np.r_[xTrain, xTest]
    else:
        c = CLASSLIST.index(LABEL)
        x = np.r_[xTrain, xTest]
        y = np.r_[yTrain, yTest].flatten()
        ilist = [i for i in range(y.shape[0]) if y[i] == c]
        xP = x[ilist]
    # NOTE: Normalize from 0 to 1 or -1 to 1
    #xP = xP/255.0
    xP = xP/127.5 - 1
    print("Shape of Preprocessed dataset: %s." % str(xP.shape))
    return xP


################################### CREATING A GAN ###################################

# Model that discriminates between fake and real dataset images
def buildDiscriminator():
    model = Sequential()

    # TODO: build a discriminator which takes in a (28 x 28 x 1) image - possibly from mnist_f
    #       and possibly from the generator - and outputs a single digit REAL (1) or FAKE (0)

    # Creating a Keras Model out of the network
  
    if DATASET != 'cifar_10':
        model.add(Flatten(input_shape = IMAGE_SHAPE))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha = alpha_relu))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha = alpha_relu))
    else:
    	model.add(Conv2D(32, kernel_size = (3,3), input_shape = IMAGE_SHAPE))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(MaxPooling2D(pool_size = (2,2)))
    	model.add(Conv2D(64, kernel_size = (3, 3)))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(MaxPooling2D(pool_size = (2,2)))
    	model.add(Flatten())
    	model.add(Dense(512))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(Dense(256))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(Dropout(0.2))
    model.add(Dense(1, activation = "sigmoid"))
    inputTensor = Input(shape = IMAGE_SHAPE)
    return Model(inputTensor, model(inputTensor))

# Model that generates a fake image from random noise
def buildGenerator():
    model = Sequential()

    # TODO: build a generator which takes in a (NOISE_SIZE) noise array and outputs a fake
    #       mnist_f (28 x 28 x 1) image

    # Creating a Keras Model out of the network

    
 
    if DATASET != 'cifar_10':
    	nodes = 128*7*7
    	model.add(Dense(nodes, input_dim=NOISE_SIZE))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(Reshape((7, 7, 128)))
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    	model.add(LeakyReLU(alpha = alpha_relu))
    	model.add(Conv2D(1, (3,3), activation='tanh', padding='same'))
    else:    

    	nodes = 256*4*4
    	model.add(Dense(nodes, input_dim=NOISE_SIZE))
    	model.add(LeakyReLU(alpha_relu))
    	model.add(Reshape((4, 4, 256)))
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    	model.add(LeakyReLU(alpha_relu))
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    	model.add(LeakyReLU(alpha_relu))
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    	model.add(LeakyReLU(alpha_relu))
    	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    model.add(Reshape(IMAGE_SHAPE))
    inputTensor = Input(shape = (NOISE_SIZE,))
    return Model(inputTensor, model(inputTensor))

def buildGAN(images, epochs = 40000, batchSize = 32, loggingInterval = 0):
    # Setup
   # opt = Adam(lr = 0.0002)
    opt = Adam(lr = 0.0002)
    loss = "binary_crossentropy"

    # Setup adversary
    adversary = buildDiscriminator()
    adversary.compile(loss = loss, optimizer = opt, metrics = ["accuracy"])

    # Setup generator and GAN
    adversary.trainable = False                     # freeze adversary's weights when training GAN
    generator = buildGenerator()                    # generator is trained within GAN in relation to adversary performance
    noise = Input(shape = (NOISE_SIZE,))
    gan = Model(noise, adversary(generator(noise))) # GAN feeds generator into adversary
    gan.compile(loss = loss, optimizer = opt)

    # Training
    trueCol = np.ones((batchSize, 1))
    falseCol = np.zeros((batchSize, 1))
    totalSteps = 0
    global USE_RATIO
    global adv_ratio
    global gen_ratio
    for epoch in range(1, epochs+1):
    	totalSteps = totalSteps + len(images)
    	if (USE_RATIO == 0 or (USE_RATIO == 1 and epoch % adv_ratio == 0)):
            # Train discriminator with a true and false batch
            batch = images[np.random.randint(0, images.shape[0], batchSize)]
            noise = np.random.normal(0, 1, (batchSize, NOISE_SIZE))
            genImages = generator.predict(noise)
            advTrueLoss = adversary.train_on_batch(batch, trueCol)
            advFalseLoss = adversary.train_on_batch(genImages, falseCol)
            advLoss = np.add(advTrueLoss, advFalseLoss) * 0.5
            adv_losses_plot[0].append(totalSteps)
            adv_losses_plot[1].append(advLoss[0])
    	if (USE_RATIO == 0 or (USE_RATIO == 1 and epoch % gen_ratio == 0)):
            # Train generator by training GAN while keeping adversary component constant
            noise = np.random.normal(0, 1, (batchSize, NOISE_SIZE))
            genLoss = gan.train_on_batch(noise, trueCol)
            gen_losses_plot[0].append(totalSteps)
            gen_losses_plot[1].append(genLoss)
            

        # Logging
    	if loggingInterval > 0 and epoch % loggingInterval == 0:
            print("\tEpoch %d:" % epoch)
            print("\t\tDiscriminator loss: %f." % advLoss[0])
            print("\t\tDiscriminator accuracy: %.2f%%." % (100 * advLoss[1]))
            print("\t\tGenerator loss: %f." % genLoss)
            runGAN(generator, OUTPUT_DIR + "/" + OUTPUT_NAME + "_test_%d.png" % (epoch / loggingInterval))
#    	if (epoch % epochs_to_view_plot == 0):
#            plt.plot(adv_losses_plot[0], adv_losses_plot[1], label="adversary")
#            plt.plot(gen_losses_plot[0], gen_losses_plot[1], label="generator")
#            #pdb.set_trace()
#            plt.legend(loc="upper left")
#            plt.show()
    	if (epoch == epochs - 1):
            plt.plot(adv_losses_plot[0], adv_losses_plot[1], label="adversary")
            plt.plot(gen_losses_plot[0], gen_losses_plot[1], label="generator")
            plt.legend(loc="upper left")
            plt.savefig(OUTPUT_DIR + "/loss.png")
    return (generator, adversary, gan)

import matplotlib.pyplot as plt
import pdb

# Generates an image using given generator
def runGAN(generator, outfile):
    noise = np.random.normal(0, 1, (1, NOISE_SIZE)) # generate a random noise array
    img = generator.predict(noise)[0]             # run generator on noise

    img = np.squeeze(img)                           # readjust image shape if needed
    #print(img)
    img = (0.5*img + 0.5)*255                       # adjust values to range from 0 to 255 as needed
    img = img.astype("uint8")
    #img = np.reshape(img, IMAGE_SHAPE)
   # plt.imshow(img)
   # plt.show()
    #imsave(outfile, img)                            # store resulting image
    if DATASET == 'cifar_10':
    	img = Image.fromarray(img, "RGB")
    else:
    	img = Image.fromarray(img, "L")
    img.save(outfile)

################################### RUNNING THE PIPELINE #############################

def main():
    print("Starting %s image generator program." % LABEL)
    # Make output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Receive all of mnist_f
    raw = getRawData()
    # Filter for just the class we are trying to generate
    data = preprocessData(raw)
    # Create and train all facets of the GAN
    (generator, adv, gan) = buildGAN(data, epochs = 10000, batchSize = 32, loggingInterval = 500)
    # Utilize our spooky neural net gimmicks to create realistic counterfeit images
    for i in range(10):
        runGAN(generator, OUTPUT_DIR + "/" + OUTPUT_NAME + "_final_%d.png" % i)
    print("Images saved in %s directory." % OUTPUT_DIR)

if __name__ == '__main__':
    main()
