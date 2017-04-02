#import the necessary packages
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras
from keras.layers import Input, merge

def SabBido_module(inp):
    
    tower_0 = Convolution2D(50, 1, 1, border_mode="same", activation='relu')(inp)

    tower_1 = Convolution2D(20, 1, 1, border_mode="same", activation='relu')(inp)
    tower_1 = Convolution2D(50, 3, 3, border_mode="same", activation='relu')(tower_1)

    tower_2 = Convolution2D(20, 1, 1, border_mode="same", activation='relu')(inp)
    tower_2 = Convolution2D(50, 5, 5, border_mode="same", activation='relu')(tower_2)    

    inception = merge([tower_0, tower_1, tower_2], mode='concat', concat_axis=1)

    return inception

def Inception_module(inp):    
    
    tower_0 = Convolution2D(20, 1, 1, border_mode="same", activation='relu')(inp)

    tower_1 = Convolution2D(20, 1, 1, border_mode="same", activation='relu')(inp)
    tower_1 = Convolution2D(20, 3, 3, border_mode="same", activation='relu')(tower_1)

    tower_2 = Convolution2D(20, 1, 1, border_mode="same", activation='relu')(inp)
    tower_2 = Convolution2D(20, 5, 5, border_mode="same", activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(inp)
    tower_3 = Convolution2D(20, 1, 1, border_mode='same', activation='relu')(tower_3)

    output = merge([tower_0, tower_1, tower_2, tower_3], mode='concat', concat_axis=1)

    return output

class LeNet:
    @staticmethod    
    def build(width, height, depth, classes, mode, weightsPath=None):
        # initialize the model
        #model = Sequential()
        inp = Input(shape=(depth, height, width))

        if mode == 1:

            print "Running MatFra Module"

            inception2 = SabBido_module(inp)
            inception = SabBido_module(inception2)
            a = Flatten()(inception)

            a = Dense(550)(a)
            #a = Dense(250)(a)
            a = Dense(classes)(a)
            out = Activation("softmax")(a)

            model = Model(input=[inp], output=[out])
        
            return model

        elif mode == 2:

            print "Running Google's Module"

            inception = Inception_module(inp)
            a = Flatten()(inception)
            a = Dense(classes)(a)
            out = Activation("softmax")(a)

            model = Model(input=[inp], output=[out])

            return model

        # first set of CONV => RELU => POOL
        #model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(depth, height, width)))
        #model.add(conv_spec(20, 5, 5, (depth, height, width), "same"))
        #model.add(Activation("relu"))
        
        # model.add(Inception_module(depth, height, width))
        # inception=Inception_module(inception)
        # second set of CONV => RELU => POOL
        # model.add(Convolution2D(50, 3, 3, border_mode="same"))
        # model.add(Activation("relu"))
            
        # set of FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(100))
        # model.add(Activation("relu"))
                # softmax classifier
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))
        
