# CS390-NIP GAN lab
# Max Jacobson / Sri Cherukuri / Anthony Niemiec
# FA2020
# uses Fashion MNIST https://www.kaggle.com/zalando-research/fashionmnist 
# uses CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html
 
import os
import numpy as np
#from scipy.misc import imsave
import random
from PIL import Image
import matplotlib.pyplot as plt
 
 
#PyTorch Imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
 
random.seed(1618)
np.random.seed(1618)
torch.manual_seed(1618)
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
# NOTE: mnist_d is no credit
# NOTE: cifar_10 is extra credit
#DATASET = "mnist_d"
DATASET = "mnist_f"
#DATASET = "cifar_10" # Not Implemented in pipeline
 
k = 1 # number of steps to apply to the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_pil_image = transforms.ToPILImage()
 
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
  
gen_losses_plot = [[], []]
adv_losses_plot = [[], []]
#epochs_to_view_plot = 5000
 
# file prefixes and directory
OUTPUT_NAME = DATASET + "_" + LABEL
OUTPUT_DIR = "./output/" + OUTPUT_NAME
 
# NOTE: switch to True in order to receive debug information
VERBOSE_OUTPUT = False
 
################################### DATA FUNCTIONS ###################################
 
# Load in and pre-process the dataset
def getRawData():
    if DATASET == "mnist_d":
        # Set up a transformation to pre-process that data
        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
        ])
        # Get the MNIST dataset
        train_data = datasets.MNIST(root='./input/data', train=True, 
            download=True, transform=transform
        )
        return train_data
    if DATASET == "mnist_f":
        # Set up a transformation to pre-process that data
        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
        ])
        # Get the MNIST Fashion dataset
        train_data = datasets.FashionMNIST(root='./input/data', train=True, 
            download=True, transform=transform
        )
        return train_data

################################### CREATING A GAN ###################################

# Generator class
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        # Sequential Adding of layers to CNN
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        # Execute the network and reshape the output
        return self.main(x).view(-1, 1, 28, 28)
 
# Discriminator Class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        # Sequential Adding of layers to CNN
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        # Flatten and execute CNN
        x = x.view(-1, 784)
        return self.main(x)
 
 
# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)
# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)
 
 
# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)
 
 
# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)
 
 
# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake, discriminator, criterion):
    b_size = data_real.size(0)  # Batch size of real data
    # Create real and fake labels
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)
    optimizer.zero_grad()  # Set optimizer for training
    output_real = discriminator(data_real)  # Discriminator output for real data
    loss_real = criterion(output_real, real_label)  # Loss for real output
    output_fake = discriminator(data_fake)  # Discriminator output for fake data
    loss_fake = criterion(output_fake, fake_label)  # Loss for fake output
    loss_real.backward()  # Backward step for real loss function
    loss_fake.backward()  # Backward step for fake loss function
    optimizer.step()  # optimizer step
    return loss_real + loss_fake  # return discriminator loss
 
 
# function to train the generator network
def train_generator(optimizer, data_fake, discriminator, criterion):
    b_size = data_fake.size(0)  # Batch size of data
    real_label = label_real(b_size)  # Create real label
    optimizer.zero_grad()  # Set optimizer for training
    output = discriminator(data_fake)  # Dsicriminator output for fake data
    loss = criterion(output, real_label)  # Get loss of generator
    loss.backward()  # Backward step for loss function
    optimizer.step()  # Optimizer step
    return loss  # return generator loss
 
def buildGAN(images, epochs = 40000, batchSize = 32, loggingInterval = 0):
    # Load data for dataset
    train_loader = DataLoader(images, batch_size=batchSize, shuffle=True)
    nz = 128
    # Create noise arrray
    noise = create_noise(64, 128)
    generator = Generator(nz).to(device)  # Get generator object
    discriminator = Discriminator().to(device)  # Get discriminator object
    gen_opt = optim.Adam(generator.parameters(), lr=0.002)  # Get Adam for generator optimizer
    dis_opt = optim.Adam(discriminator.parameters(), lr=0.002)  # Get Adam for discriminator optimizer
    criterion = nn.BCELoss()  # Use Binary Cross-Entropy loss function
    images = []
    generator.train()  # Set generator to train
    discriminator.train()  # Set discriminator to train
    for epoch in range(epochs):
        gen_loss = 0.0
        dis_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), total=int(len(images)/train_loader.batch_size)):
            img, _ = data  # Get image
            img = img.to(device)
            b_size = len(img)  # Get number of images
            # Used for training ratio based on k
            for step in range(k):
                # Generate fake image and calculate discriminator loss
                fake = generator(create_noise(b_size, nz)).detach()
                real = img
                dis_loss += train_discriminator(dis_opt, real, fake, discriminator, criterion)
            # Get fake image and calculate generator loss
            fake = generator(create_noise(b_size, nz))
            gen_loss += train_generator(gen_opt, fake, discriminator, criterion)
        # Generate an image at the end of the epoch and print loss for epoch
        gen_img = generator(noise).cpu().detach()
        gen_img = make_grid(gen_img)
        save_generator_image(gen_img, f"./output/gen_img{epoch}.png")
        images.append(gen_img)
        gen_epoch_loss = gen_loss / i
        dis_epoch_loss = dis_loss / i
 
        print(f"Epoch {epoch} of {epochs}")
        print(f"Generator loss: {gen_epoch_loss:.8f}\nDiscriminator loss: {dis_epoch_loss:.8f}")
    imgs = [np.array(to_pil_image(image)) for image in images]
    imageio.mimsave('./output/generator_images.gif')
    return generator
 
 
import matplotlib.pyplot as plt
import pdb
 
# Generates an image using given generator
def runGAN(generator, outfile):
    noise = create_noise(sample_size, nz)
    img = generator(noise).cpu().detach()
    save_generator_image(img, outfile)
 
 
################################### RUNNING THE PIPELINE #############################
 
def main():
    print("Starting %s image generator program." % LABEL)
    # Make output directory
    if not os.path.exists(OUTPUT_DIR):  # Unused DIR
        os.makedirs(OUTPUT_DIR)
    # Receive all of mnist_f
    raw = getRawData()
    # Filter for just the class we are trying to generate
    data = raw
    # Create and train all facets of the GAN
    generator = buildGAN(data, epochs = 10000, batchSize = 512, loggingInterval = 500)
    # Utilize our spooky neural net gimmicks to create realistic counterfeit images
    for i in range(10):
        runGAN(generator, OUTPUT_DIR + "/" + OUTPUT_NAME + "_final_%d.png" % i)
    print("Images saved in %s directory." % OUTPUT_DIR)
 
if __name__ == '__main__':
    main()