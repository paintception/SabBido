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
    
    #tower_0 = Convolution2D(7, 1, 1, border_mode="same", activation='relu')(inp)

    tower_1 = Convolution2D(2, 1, 1, border_mode="same", activation='relu')(inp)
    tower_1 = Convolution2D(4, 3, 3, border_mode="same", activation='relu')(tower_1)

    tower_2 = Convolution2D(2, 1, 1, border_mode="same", activation='relu')(inp)
    tower_2 = Convolution2D(4, 5, 5, border_mode="same", activation='relu')(tower_2)    

    #inception = merge([tower_0, tower_1, tower_2], mode='concat', concat_axis=1)
    inception = merge([ tower_1, tower_2], mode='concat', concat_axis=1)

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


def build(width, height, depth, classes, weightsPath=None):
    inp = Input(shape=(height, width, depth))

    lrelu = keras.layers.advanced_activations.LeakyReLU(alpha=0.03)

    print "Running MatFra Module"

    tower_1 = Convolution2D(20, kernel_size=(7), strides=(3), border_mode="valid", activation=lrelu)(inp)
    tower_1 = Convolution2D(18, kernel_size=(9), strides=(2), border_mode="valid", activation=lrelu)(tower_1)
    tower_1 = Convolution2D(10, kernel_size=(11), border_mode="valid", activation=lrelu)(tower_1)
    tower_1 = Convolution2D(8, kernel_size=(22), border_mode="valid", activation=lrelu)(tower_1)
    #tower_1 = Convolution2D(32, kernel_size=(33), border_mode="valid", activation=lrelu)(tower_1)
    inception2 = SabBido_module(tower_1)
    #inception = SabBido_module(inception2)
    a = Flatten()(inception2)

    a = Dense(100)(a)
    #a = Dense(250)(a)
    a = Dense(classes)(a)
    out = Activation("softmax")(a)

    model = Model(input=[inp], output=[out])

    return model



def MatFra_plots(history, i):

    print "Storing SabBido's Results for experiment: ", i

    f1 = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./res/MatFraAccuracyExp_'+str(i))

    #np.save('./res/MatFraTrainingAccuracyExp_'+str(i), np.asarray(history.history['acc']))
    #np.save('./res/MatFraValidationAccuracyExp_'+str(i), np.asarray(history.history['val_acc']))

    f2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./res/MatFraLossExp_'+str(i))
    
    #np.save('./res/MatFraAccuracyLossExp_'+str(i), np.asarray(history.history['loss']))
    #np.save('./res/MatFraValidationLossExp_'+str(i), np.asarray(history.history['val_loss']))

def Google_plots(history, i):

    print "Storing Google Results for experiment: ", i

    f1 = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./Plots/GoogleAccuracyExp_'+str(i))
    plt.close()

    np.save('./res/GoogleTrainingAccuracyExp_'+str(i), np.asarray(history.history['acc']))
    np.save('./res/GoogleValidationAccuracyExp_'+str(i), np.asarray(history.history['val_acc']))

    f2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./Plots/GoogleLossExp_'+str(i))
    plt.close()

    np.save('./res/GoogleAccuracyLossExp_'+str(i), np.asarray(history.history['loss']))
    np.save('./res/GoogleValidationLossExp_'+str(i), np.asarray(history.history['val_loss']))


