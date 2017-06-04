#import the necessary packages
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout, Reshape
import keras
from keras.layers import Input, merge



class LeNet:
    @staticmethod    
    def build(width, height, depth, classes, mode, weightsPath=None):
        # initialize the model
        #model = Sequential()
        inp = Input(shape=(height, width, depth))
    
        lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.03)

        if mode == 1:

            print "Running Autoencoder"

            tower_1 = Convolution2D(24, kernel_size=(5), strides=(2), border_mode="same", activation='relu')(inp)
            tower_1 = Convolution2D(48, kernel_size=(3), strides=(3), border_mode="same", activation='relu')(tower_1)
            tower_1 = Convolution2D(32, kernel_size=(5), strides=(2), border_mode="same", activation='relu')(tower_1)
            tower_1 = Convolution2D(24, kernel_size=(7), border_mode="same", activation='relu')(tower_1)
            tower_1 = Convolution2D(24, kernel_size=(9), border_mode="same", activation='relu')(tower_1)
            tower_1 = Convolution2D(24, kernel_size=(5), strides=(5), border_mode="same", activation='relu')(tower_1)

            encoded = Flatten()(tower_1)
            #a = Dropout(0.3)(a)
            encoded = Dense(5*5*24, activation='relu')(encoded)

            decoded = Dense(5*5*24, activation='relu')(encoded)
            decoded = Reshape((5,5,24))(decoded)

            tower_2 = Deconvolution2D(24, kernel_size=(5), strides=(5), border_mode="same", activation='relu')(decoded)
            tower_2 = Deconvolution2D(24, kernel_size=(9), border_mode="same", activation='relu')(tower_2)            
            tower_2 = Deconvolution2D(24, kernel_size=(7), border_mode="same", activation='relu')(tower_2)
            tower_2 = Deconvolution2D(32, kernel_size=(5), strides=(2), border_mode="same", activation='relu')(tower_2)
            tower_2 = Deconvolution2D(48, kernel_size=(3), strides=(3), border_mode="same", activation='relu')(tower_2)
            tower_2 = Deconvolution2D(24, kernel_size=(5), strides=(2), border_mode="same", activation='relu')(tower_2)
            recustructed = Deconvolution2D(3, kernel_size=(3), border_mode="same", activation='relu')(tower_2)

            model = Model(input=[inp], output=[recustructed])
            #model = Model(input=[inp], output=[encoded])
        
            return model


        
