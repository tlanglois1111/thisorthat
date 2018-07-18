# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 100
INIT_LR = 1e-3
batch_size = 10

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "cats" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=100, height=100, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
                        epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Cats/Not Cats")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


"""
On Lines 2-18 we import required packages. There packages enable us to:

Load our image dataset from disk
Pre-process the images
Instantiate our Convolutional Neural Network
Train our image classifier
Notice that on Line 3 we set the matplotlib  backend to "Agg"  so that we can save the plot to disk in the background. 
This is important if you are using a headless server to train your network (such as an Azure, AWS, or other cloud instance).

From there, we parse command line arguments:

Here we have two required command line arguments, --dataset  and --model , as well as an optional path to our accuracy/loss chart, --plot .

The --dataset  switch should point to the directory containing the images we will be training our image classifier on 
(i.e., the “Cats” and “Not Cats” images) while the --model  switch controls where we will save our serialized image classifier after it has been trained. 
If --plot  is left unspecified, it will default to plot.png  in this directory if unspecified.

Next, we’ll set some training variables, initialize lists, and gather paths to images:

On Lines 32-34 we define the number of training epochs, initial learning rate, and batch size.

Then we initialize data and label lists (Lines 38 and 39). 
These lists will be responsible for storing our the images we load from disk along with their respective class labels.

From there we grab the paths to our input images followed by shuffling them (Lines 42-44).

Now let’s pre-process the images:

This loop simply loads and resizes each image to a fixed 28×28 pixels (the spatial dimensions required for LeNet), 
and appends the image array to the data  list (Lines 49-52) followed by extracting the class label  from the imagePath  on Lines 56-58.

I prefer organizing deep learning image datasets in this manner as it allows us to efficiently organize our dataset 
and parse out class labels without having to use a separate index/lookup file.

Next, we’ll scale images and create the training and testing splits:

On Line 61 we further pre-process our input data by scaling the data points from [0, 255] (the minimum and maximum RGB values of the image) to the range [0, 1].

We then perform a training/testing split on the data using 75% of the images for training and 25% for testing (Lines 66 and 67). 
This is a typical split for this amount of data.

We also convert labels to vectors using one-hot encoding — this is handled on Lines 70 and 71.

Subsequently, we’ll perform some data augmentation, enabling us to generate “additional” training data 
by randomly transforming the input images using the parameters below:

Data augmentation is covered in depth in the Practitioner Bundle of my new book, Deep Learning for Computer Vision with Python.

Essentially Lines 74-76 create an image generator object which performs random rotations, shifts, flips, crops, and sheers on our image dataset. 
This allows us to use a smaller dataset and still achieve high results.

Let’s move on to training our image classifier using deep learning and Keras.

We’ve elected to use LeNet for this project for two reasons:

LeNet is a small Convolutional Neural Network that is easy for beginners to understand
We can easily train LeNet on our Cats/Not Cats dataset without having to use a GPU
If you want to study deep learning in more depth (including ResNet, GoogLeNet, SqueezeNet, and others) 
please take a look at my book, Deep Learning for Computer Vision with Python.
We build our LeNet model along with the Adam  optimizer on Lines 80-83. Since this is a two-class classification problem 
we’ll want to use binary cross-entropy as our loss function. If you are performing classification with > 2 classes, 
be sure to swap out the loss  for categorical_crossentropy .

Training our network is initiated on Lines 87-89 where we call model.fit_generator , 
supplying our data augmentation object, training/testing data, and the number of epochs we wish to train for.

Line 93 handles serializing the model to disk so we later use our image classification without having to retrain it.

Finally, let’s plot the results and see how our deep learning image classifier performed:


"""