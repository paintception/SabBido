import numpy as np
import argparse
#import cv2
#import seaborn
import keras
import os
import time
import tensorflow as tf 

from pyimagesearch.cnn.networks import LeNet
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def shape_data(dataset):
    
    dataset = np.reshape(dataset, (dataset.shape[0], 1, 8, 8))

    return dataset

def make_categorical(labels, n_classes):
    return(np_utils.to_categorical(labels, n_classes))

def MatFra_plots(history, i):

    print "Storing SabBido's Results for experiment: ", i

    f1 = plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('../res/MatFraAccuracyExp_'+str(i))

    #np.save('./res/MatFraTrainingAccuracyExp_'+str(i), np.asarray(history.history['acc']))
    #np.save('./res/MatFraValidationAccuracyExp_'+str(i), np.asarray(history.history['val_acc']))

    f2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('../res/MatFraLossExp_'+str(i))
    
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
    plt.savefig('../Plots/GoogleAccuracyExp_'+str(i))
    plt.close()

    np.save('../res/GoogleTrainingAccuracyExp_'+str(i), np.asarray(history.history['acc']))
    np.save('../res/GoogleValidationAccuracyExp_'+str(i), np.asarray(history.history['val_acc']))

    f2 = plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('./Plots/GoogleLossExp_'+str(i))
    plt.close()

    np.save('../res/GoogleAccuracyLossExp_'+str(i), np.asarray(history.history['loss']))
    np.save('../res/GoogleValidationLossExp_'+str(i), np.asarray(history.history['val_loss']))

def final_prediction_test():    
    for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
        # classify the digit
        probs = model.predict(testData[np.newaxis, i])
        prediction = probs.argmax(axis=1)

        # resize the image from a 28 x 28 image to a 96 x 96 image so we
        # can better see it
        #image = (testData[i][0] * 255).astype("uint8")
        #image = cv2.merge([image] * 3)
        #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        #cv2.putText(image, str(prediction[0]), (5, 20),
        #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # show the image and prediction
        print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
            np.argmax(testLabels[i])))
        #cv2.imshow("Digit", image)
        #cv2.waitKey(0)

def main():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
    args = vars(ap.parse_args())

    #Param Configuration!    

    path = '/home/borg/SabBido/DataSet/'
    pic_shape = (300,300,1)
    n_epochs = 25
    opt = SGD(lr=0.001)
    n_classes = 21
    train = False  # If false, load parameters and run validation!

    precise_evaluation = False
    ##################################
   
    trainData = np.load(path+'x.npy')
    trainData = np.reshape(trainData,(trainData.shape[0], pic_shape[0],pic_shape[1],pic_shape[2]))
    trainLabels = np.load(path+'y.npy')

    testData = np.load(path+'x.npy')
    testData = np.reshape(testData, (testData.shape[0], pic_shape[0],pic_shape[1],pic_shape[2]))
    testLabels = np.load(path+'y.npy')
    print 'Data Loaded!'
    
    




    print 'Deleting old logs in 3 sec...'
    time.sleep(3)
    items = os.listdir('/home/borg/SabBido/logs')
    [os.remove('/home/borg/SabBido/logs/'+i ) for i in items] 


    print "Running Experiment: ", i


    tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/borg/SabBido/logs', 
                            histogram_freq=0, write_graph=True, write_images=False)

    print("[INFO] compiling model...")

    model_MatFra = LeNet.build(pic_shape[0],pic_shape[1],pic_shape[2], classes=n_classes,
             mode=1, weightsPath=args["weights"] if args["load_model"] > 0 else None)
    model_MatFra.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print model_MatFra.summary()
    time.sleep(5)  

    if train:
        history_MatFra = model_MatFra.fit(trainData, trainLabels, batch_size=50,
                 nb_epoch=n_epochs, verbose=1, validation_data=(testData, testLabels),
                 callbacks=[tbCallBack])

        # serialize model to JSON
        #model_json = model_MatFra.to_json()
        #with open("../NN_param/model.json", "w") as json_file:
        #    json_file.write(model_json)

        #serialize weights to HDF5
        model_MatFra.save_weights("../NN_param/model.h5")
        print("Saved model to disk")

    else:

        model_MatFra.load_weights("../NN_param/model.h5")
        print("[INFO] evaluating...")
    (loss, accuracy) = model_MatFra.evaluate(testData, testLabels, batch_size=50, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    if precise_evaluation:
        print 'specific: '
        for i in xrange(len(testData)):
        # classify the digit
            probs = model_MatFra.predict(testData[np.newaxis, i])
            prediction = probs.argmax(axis=1)
            print "#########################"
            print testLabels[i] 
            probs = np.round(probs, decimals = 2)           
            print probs
            print "-------------------------"
            time.sleep(0.5)

      

if __name__ == '__main__':
    main()
