import numpy as np
np.random.seed(42)
import argparse
import cv2
import random
#import seaborn
import keras
import os
import time
import math
from PIL import Image
import tensorflow as tf 
from keras.callbacks import EarlyStopping

from GenerativeNN import * 
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def randomize(inp, out):

    n_inp=[]
    n_out=[]
    indxs=[i for i in xrange(len(inp))]
    random.shuffle(indxs)
    for ind in indxs:

        n_inp.append(inp[ind])
        n_out.append(out[ind])

    return(np.asarray(n_inp), np.asarray(n_out))

def resize_pic(pic, shape = (64,64)):
    pic = pic *255
    pic = np.asarray(pic, dtype = np.uint8)
    pic = cv2.resize(pic, shape)
    pic = np.asarray(pic, dtype = np.float32)
    pic = pic / 255.
    return pic



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]

    return image



def main():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

   

    #Param Configuration!    

    path = '/home/borg/SabBido/DataSet/'
    #path = '/home/borg/sudoRepo/Thesis/DataSet/'    
    pic_shape = (128,128,3)
    n_epochs = 10000
    n_batch = 25
    opt = SGD(lr=0.0001)
    adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.99, epsilon=1e-08, decay=0.0)
    
    train = True  # If false, load parameters and run validation!

    precise_evaluation = False
    ##################################
    X_train = np.load(path+'bido_128.npy')

    #tmp = [resize_pic(pic) for pic in X_train]
    #X_train = np.asarray(tmp)
    #trainData = trainData*2 -1

    #X_train, X_test, y_train,  y_test = train_test_split(
    #trainData, trainData, test_size=0.02, random_state=0)

    #testData = np.load(path+'test_x.npy')
    #testData = np.reshape(testData, (testData.shape[0], pic_shape[0],pic_shape[1],pic_shape[2]))
    #testLabels = np.load(path+'test_y.npy')
    
    print 'Data Loaded!'
    
    




    print 'Deleting old logs in 3 sec...'
    time.sleep(1)
    items = os.listdir('/home/borg/SabBido/logs')
    [os.remove('/home/borg/SabBido/logs/'+i ) for i in items] 



    tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/borg/SabBido/logs', 
                            histogram_freq=0, write_graph=True, write_images=True)

    print("[INFO] compiling model...")

    discriminator = discriminator_model()
    #discriminator.load_weights('discriminator')
    generator = generator_model()
    #generator.load_weights('generator')

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    generator.compile(loss='logarimic_crossentropy', optimizer=opt)
    discriminator.trainable = False

    discriminator_on_generator.compile(loss='logarimic_crossentropy', optimizer=opt)

    discriminator.trainable = True
    discriminator.compile(loss='logarimic_crossentropy', optimizer=opt)

    

    print 'generator'
    print generator.summary()
    print 'discriminator'
    print discriminator.summary()
    print '//////////////////////////'
    print 'discriminator_on_generator'
    print discriminator_on_generator.summary()
    time.sleep(1)

    noise = np.zeros((n_batch, 100))

    # Training part!
    for epoch in xrange(n_epochs):

        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/n_batch))
        for index in range(int(X_train.shape[0]/n_batch)):

            for i in range(n_batch):
                noise[i, :] = np.random.uniform(0, 1, 100)
            #print 'noise is: ', noise

            image_batch = X_train[index*n_batch:(index+1)*n_batch]
            generated_images = generator.predict(noise, verbose=0)

            
            if index % 500 == 0:
                print 'min_real: ', image_batch.min()
                print 'max_real: ', image_batch.max()
                print 'min: ', generated_images.min()
                print 'max: ', generated_images.max()
                print 'shape: ', generated_images.shape
                print 'shape pic : ', generated_images[0].shape
                path ='./ris3/'
                #try:
                #    os.mkdir(path)
                #except:
                #    pass
                #for indd,pic in enumerate(generated_images):
                #pic = (pic+1)/2.
                pic = generated_images[0]    
                pic = pic *255
                pic = np.asarray(pic, dtype = np.uint8)
                cv2.imwrite(path+'/pic_{}_{}.png'.format(epoch,index), pic)
                cv2.imshow('real', cv2.resize(image_batch[0],(600,600)))
                cv2.imshow('gene', cv2.resize(generated_images[0],(600,600)))
                cv2.waitKey(200)
                

            X = np.concatenate((image_batch, generated_images))
            y = [0.9] * n_batch + [0] * n_batch
            X,y=randomize(X,y)
            d_loss = discriminator.train_on_batch(X, y)
            
            #for i in range(n_batch):
            #    noise[i, :] = np.random.uniform(0, 1, 100)
            #discriminator.trainable = False
            
            #if index %2 ==0:
            g_loss = discriminator_on_generator.train_on_batch(
                                noise, [0.9] * n_batch)
            #discriminator.trainable = True
            if index % 30 == 0:
                print("batch %d g_loss : %f" % (index, g_loss))
                print("batch %d d_loss : %f" % (index, d_loss))
            '''
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(n_batch):
                noise[i, :] = np.random.uniform(0, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * n_batch)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            '''
            #if index % 10 == 9:
        generator.save_weights('generator', True)
        discriminator.save_weights('discriminator', True)

        
            


    '''
    #model_MatFra.save_weights("../NN_param_autoencoder/startingWeights.h5")
    
    print model_MatFra.summary()
    time.sleep(2)  

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    ind = 0

    if train:

        try:
            model_MatFra.load_weights("../NN_param_autoencoder_bido/model.h5")
            print 'previous weight loaded'
        except:
            print 'No trained weight'

        history_MatFra = model_MatFra.fit(trainData, trainLabels, n_batch=30,
                 epochs=n_epochs, verbose=1, validation_data=(testData, testLabels),
                 callbacks=[tbCallBack])

            # serialize model to JSON
            #model_json = model_MatFra.to_json()
            #with open("../NN_param/model.json", "w") as json_file:
            #    json_file.write(model_json)

            #serialize weights to HDF5
        model_MatFra.save_weights("../NN_param_autoencoder_bido/model.h5")
        model_MatFra.save_weights("../NN_param_autoencoder_bido/model_{}.h5".format(ind+1))
        print("Saved model to disk")

    else:

        model_MatFra.load_weights("../NN_param_autoencoder_bido/model.h5")
        print("[INFO] evaluating...")
    #accuracy = model_MatFra.evaluate(testData, testLabels, n_batch=50, verbose=1)
    #print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


    if precise_evaluation:
        print 'specific: '
        #for ind in xrange(100):
        ind =0
        #    print 'using weight ', ind
        #    model_MatFra.load_weights("../NN_param_autoencoder/model_{}.h5".format(ind))
        for i in xrange(10):
            #i = 134
        # classify the digit
            #pic = cv2.imread('/home/borg/SabBido/FaceDetect/Data/20160716_111233.jpg')
            #pic = cv2.resize(pic, (150,150))
            #pic = np.asarray(pic, dtype = np.float32)            
            #pic = pic/255
            
            #orig = pic
            orig = testData[i]
            orig = orig *255
            orig = np.array(orig, dtype = np.uint8)
            probs = model_MatFra.predict(testData[np.newaxis, i])
            #pic = pic.reshape((1,150,150,3))  
            #probs = model_MatFra.predict(pic)
        
            probs = probs *255
            probs = np.array(probs, dtype = np.uint8)
            probs = probs.reshape(pic_shape)
            cv2.imwrite('/home/borg/SabBido/test/pic_{}_{}.jpg'.format(ind, i), probs)
            cv2.imwrite('/home/borg/SabBido/test/original.jpg'.format(ind, i), orig)
            cv2.imshow('original',testData[i])
            cv2.imshow('recunstructed', probs)
            cv2.waitKey(1) 
            time.sleep(1.5)
    '''
      

if __name__ == '__main__':
    main()
