# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

"""
Lines 2-8 handle importing our required Python packages. The Conv2D  class is responsible for performing convolution. 
We can use the MaxPooling2D  class for max-pooling operations. 
As the name suggests, the Activation  class applies a particular activation function. 
When we are ready to Flatten  our network topology into fully-connected, Dense  layer(s) we can use the respective class names.

The LeNet  class is defined on Line 10 followed by the build  method on Line 12. 
Whenever I defined a new Convolutional Neural Network architecture I like to:

Place it in its own class (for namespace and organizational purposes)
Create a static build  function that builds the architecture itself
The build  method, as the name suggests, takes a number of parameters, each of which I discuss below:

width : The width of our input images
height : The height of the input images
depth : The number of channels in our input images ( 
    1  for grayscale single channel images, 
    3  for standard RGB images which we’ll be using in this tutorial)
classes : The total number of classes we want to recognize (in this case, two)
We define our model  on Line 14. We use the Sequential  class since we will be sequentially adding layers to the model .

Line 15 initializes our inputShape  using channels last ordering (the default for TensorFlow). 
If you are using Theano (or any other backend to Keras that assumes channels first ordering), 
Lines 18 and 19 properly update the inputShape .

Now that we have initialized our model, we can start adding layers to it:

The CONV  layer will learn 20 convolution filters, each of which are 5×5.

We then apply a ReLU activation function followed by 2×2 max-pooling in both the x and y direction with a stride of two. 
To visualize this operation, consider a sliding window that “slides” across the activation volume, 
taking the max operation over each region, while taking a step of two pixels in both the horizontal and vertical direction.

Let’s define our second set of CONV => RELU => POOL  layers:

This time we are learning 50 convolutional filters rather than the 20 convolutional filters as in the previous layer set. 
It’s common to see the number of CONV  filters learned increase the deeper we go in the network architecture.

Our final code block handles flattening out the volume into a set of fully-connected layers:

On Line 33 we take the output of the preceding MaxPooling2D  layer and flatten it into a single vector. 
This operation allows us to apply our dense/fully-connected layers.

Our fully-connected layer contains 500 nodes (Line 34) which we then pass through another nonlinear ReLU activation.

Line 38 defines another fully-connected layer, 
but this one is special — the number of nodes is equal to the number of classes  (i.e., the classes we want to recognize).

This Dense  layer is then fed into our softmax classifier which will yield the probability for each class.

Finally, Line 42 returns our fully constructed deep learning + Keras image classifier to the calling function.
"""
