import numpy as np
import random
import argparse
import cv2
import seaborn
import keras
import os
import time
import tensorflow as tf 

from NN_constructor import *
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


class batch_handler(object):
    def __init__(self, path, test_size = 10, batch_size = 50, data_shape = (300,300,1)):
        
        self.loaded_all = False
        self.in_ram = False

        self.test_size = test_size
        self.path = path
        self.batch_size = batch_size
        self.data_shape = data_shape

        self.classes = os.listdir(self.path)
        self.classes.sort()
        self.n_classes = len(self.classes)
        print "We have "+str(self.n_classes)+" classes", 

        (self.pics, self.labels) = self.get_all_pic_path()
        print "Picture path analyzed. Total data: ", len(self.pics)
        
        self.randomize_order()
        self.split_train_test()
        print "Test set generated"

        self.num_data = len(self.pics)
        self.devide_in_batches()
        print "Train data: ", self.num_data
        print "Dived in n_batches: ", self.n_batch

        


        
    def get_train_batch(self, ind):
        
        if not self.in_ram:    
            
            labels = np.asarray(self.batches_labe[ind])
            pics_path = self.batches_pics[ind]
            data = [self.__load_gray_pic_reshape(pic) for pic in pics_path]
            data = np.asarray(data, dtype = np.float32)
            data = data/255.
            data = np.reshape(data, (len(data), self.data_shape[0], 
                                    self.data_shape[1], self.data_shape[2]))
        
        if self.loaded_all:
            
            if not self.in_ram:
               
                self.data = data
                self.labels = labels
                self.in_ram = True
                print " All dataset in ram!"

            return (self.data, self.labels)
                
        
        return (data, labels)


    def generator_train_batch():
        
        for batch in xrange(n_batch):

            data, labels = self.get_train_batch(batch)



    def get_test_batch(self, ind):

        labels = np.asarray(self.batches_test_labe[ind])
        pics_path = self.batches_test_pics[ind]
        data = [self.__load_gray_pic_reshape(pic) for pic in pics_path]
        data = np.asarray(data)
        data = np.reshape(data, (len(data), self.data_shape[0], 
                                self.data_shape[1], self.data_shape[2]))
        
        return (data, labels)


    def __load_gray_pic_reshape(self, pic):
        try:
            img = cv2.imread(pic, 0)
            img = cv2.resize(img, (self.data_shape[0],self.data_shape[1]))
        except:
            print 'Error with pic', pic
            os.remove(pic)
            time.sleep(1)

        return img


    def rearrange_data(self):
        
        (self.pics, self.labels) = self.get_all_pic_path()
        self.randomize_order()
        self.split_train_test()
        self.num_data = len(self.pics)
        self.devide_in_batches()    


    def devide_in_batches(self):

        left = self.num_data % self.batch_size
        self.n_batch = self.num_data / self.batch_size
        

        if self.num_data < self.batch_size:

            self.loaded_all = True
            
            
        self.batches_pics = []
        self.batches_labe = []
        cnt = 0
        for i in xrange(self.n_batch):

            self.batches_pics.append(self.pics[cnt:cnt+self.batch_size])
            self.batches_labe.append(self.labels[cnt:cnt+self.batch_size])
            cnt += self.batch_size
        
        if left != 0:
            self.n_batch += 1
            self.batches_pics.append(self.pics[cnt:])
            self.batches_labe.append(self.labels[cnt:])

        ########################## Now the test ################################
        left = len(self.test_pics) % self.batch_size
        self.n_test_batch = len(self.test_pics) / self.batch_size
        self.batches_test_pics = []
        self.batches_test_labe = []
        cnt = 0
        for i in xrange(self.n_test_batch):

            self.batches_test_pics.append(self.test_pics[cnt:cnt+self.batch_size])
            self.batches_test_labe.append(self.test_labels[cnt:cnt+self.batch_size])
            cnt += self.batch_size
        
        if left != 0:
            self.n_test_batch += 1
            self.batches_test_pics.append(self.test_pics[cnt:])
            self.batches_test_labe.append(self.test_labels[cnt:])
    
         

    def get_all_pic_path(self):

        all_pic_path = []
        all_pic_labe = []
        for fold in self.classes:

            label = [0 for i in xrange(self.n_classes)]
            label[self.classes.index(fold)] = 1
            tmp_path = self.path + fold + '/'
            pics = os.listdir(tmp_path)
            for pic in pics:

                pic_path = tmp_path + pic
                all_pic_path.append(pic_path)
                all_pic_labe.append(label)
         
         
        return (all_pic_path, all_pic_labe)


    def randomize_order(self):

        index = [i for i in xrange(len(self.pics))]
        random.shuffle(index)
        n_pics = [self.pics[i] for i in index]
        n_labels = [self.labels[i] for i in index]
        self.pics = n_pics
        self.labels = n_labels

    
    def split_train_test(self):

        train_index = len(self.pics) / self.test_size
        self.test_pics = self.pics[:train_index]
        self.pics = self.pics[train_index:] 
        self.test_labels = self.labels[:train_index]
        self.labels = self.labels[train_index:]        


def main():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

   
    #################################
    #      Param Configuration!     #
    #################################
    path = '/home/borg/SabBido/Images/'
    pic_shape = (300,300,1)
    n_epochs = 100
    batch_size = 16000 # This is dataset load on ram. If possible will save all data in ram
    batch_size2 = 50  # This is dataset load on Video ram. This has to be smaller than prev one!!!!!!!!!!!!! 
    opt = SGD(lr=0.001)
    train = True                   # If false, load parameters and run validation!
    load = True                    # load from a previous stage 
    if not load:
        print 'overwriting parameters in 5 sec...'
        time.sleep(5)
    precise_evaluation = False
    ##################################

    

    #print 'Deleting old logs in 3 sec...'
    #time.sleep(3)
    items = os.listdir('/home/borg/SabBido/logs')
    [os.remove('/home/borg/SabBido/logs/'+i ) for i in items] 

    tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/borg/SabBido/logs', 
                            histogram_freq=0, write_graph=True, write_images=False)

    print("[INFO] compiling model...")

    DataSet = batch_handler(path, test_size = 10, batch_size = batch_size, data_shape = pic_shape)
    n_classes = DataSet.n_classes
    time.sleep(1)

    model_MatFra = build(pic_shape[0],pic_shape[1],pic_shape[2], classes=n_classes)
    model_MatFra.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print model_MatFra.summary()
    time.sleep(2)  

    if load:
        try:
            model_MatFra.load_weights("./NN_param/model.h5")
            print("Param Loaded!")
        except:
            print "No model found"


    if train:
        
        n_batch = DataSet.n_batch
        n_test_batch = DataSet.n_test_batch
        
        #for epoch in xrange(n_epochs):
            
            #for batch in xrange(n_batch):
                #(x,y) = DataSet.get_train_batch(batch)
                
                #history_MatFra = model_MatFra.fit(x,y, batch_size=batch_size2,
                #         verbose = 1, callbacks=[tbCallBack])
        history_MatFra = model_MatFra.fit_generator(DataSet.get_train_batch(),
                    batch_size=batch_size2, verbose = 1, callbacks=[tbCallBack])

        print "Epoch: {}; Batch: {} / {} ".format(epoch, batch, n_batch)
                '''
                for x_in, y_in in zip(x,y):
                    s = x_in.shape 
                    x_in = np.reshape(x_in, (1, s[0],s[1],s[2]))
                    s = y_in.shape
                    y_in = np.reshape(y_in, (1, s[0]))
                    model_MatFra.train_on_batch(x_in, y_in)
                '''
            if epoch%5 == 0:
                #serialize weights to HDF5
                accuracy = []
                loss = []
                for test_batch in xrange(n_test_batch):
                    
                    print 'test_batch: {}, n_test_batch: {}'.format(test_batch,n_test_batch)
                    (test_x, test_y) = DataSet.get_test_batch(test_batch)
                    print "test x : {}, test y: {}".format(len(test_x), len(test_y))

                    (tmp_loss, tmp_accuracy) = model_MatFra.evaluate(test_x, test_y,
                                                             batch_size=batch_size2)
                    accuracy.append(tmp_accuracy)
                    loss.append(tmp_loss)

                loss = np.asarray(loss).mean()
                accuracy = np.asarray(accuracy).mean()
                print("\n Val_accuracy: {}; Val_loss: {}".format(accuracy * 100, loss))
                
                model_MatFra.save_weights("./NN_param/model.h5")
                print("Saved model to disk")
        #serialize weights to HDF5
        model_MatFra.save_weights("./NN_param/model.h5")
        print("Saved model to disk")

    


    
    accuracy = []
    loss = []
    for test_batch in xrange(n_test_batch):

        (test_x, test_y) = DataSet.get_test_batch(test_batch)
        (tmp_loss, tmp_accuracy) = model_MatFra.evaluate(test_x, test_y, batch_size=batch_size2)
        accuracy.append(tmp_accuracy)
        loss.append(tmp_loss)

    loss = np.asarray(loss).mean()
    accuracy = np.asarray(accuracy).mean()
    print("Val_accuracy: {}; Val_loss: {}".format(accuracy * 100, loss))



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
