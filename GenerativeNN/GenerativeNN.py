#import the necessary packages
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout, Reshape
import keras
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization






def generator_model():

    
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024*4*4,kernel_initializer='random_uniform'))
    #model.add(BatchNormalization())  
    model.add(Reshape((4,4,1024))) 
    model.add(Deconvolution2D(512, kernel_size=(3), strides=(2), border_mode="same", activation='relu', kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Deconvolution2D(256, kernel_size=(5), strides=(2), border_mode="same", activation='relu', kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Deconvolution2D(128, kernel_size=(5), strides=(2), border_mode="same", activation='relu', kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Deconvolution2D(64, kernel_size=(5), strides=(2), border_mode="same", activation='relu', kernel_initializer='random_uniform'))
    model.add(BatchNormalization())
    model.add(Deconvolution2D(3 , kernel_size=(5), strides=(2), border_mode="same", activation='sigmoid', kernel_initializer='random_uniform'))
    #model.add(BatchNormalization(axis = 1))

    return model


def discriminator_model():
    keras.initializers.Initializer()
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
    lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.03)
    model = Sequential()
    model.add(Convolution2D(64, input_shape=(128,128,3), kernel_size=(5), strides=(2), border_mode="same", activation='relu'))
    model.add(Convolution2D(128, kernel_size=(5), strides=(2), border_mode="same", activation=lrelu))
    #model.add(BatchNormalization(axis = 1))
    model.add(Convolution2D(256, kernel_size=(5), strides=(2), border_mode="same", activation=lrelu))
    #model.add(BatchNormalization(axis = 1))
    model.add(Convolution2D(512, kernel_size=(5), strides=(2), border_mode="same", activation=lrelu))
    #model.add(BatchNormalization(axis = 1))
    model.add(Convolution2D(1024, kernel_size=(5), strides=(2), border_mode="same", activation=lrelu))
    #model.add(BatchNormalization(axis = 1))
    model.add(Flatten())
    model.add(Dense(2, activation = 'softmax'))

    return model


def generator_containing_discriminator(generator, discriminator):

    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)

    return model



        
