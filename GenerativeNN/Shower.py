import numpy as np
np.random.seed(42)
import argparse
import cv2
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


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]

        for ind, pic in enumerate(nice_images):
            pic = pic*255
            pic = np.asarray(pic, dtype = np.uint8)
            cv2.imwrite('./test/pic_{}.png'.format(ind), pic)
        #image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(0, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        for ind, pic in enumerate(generated_images):
            pic = pic*255
            pic = np.asarray(pic, dtype = np.uint8)
            cv2.imwrite('./test/pic_{}.png'.format(ind), pic)

        #image = combine_images(generated_images)
    #image = image*127.5+127.5
    #Image.fromarray(image.astype(np.uint8)).save(
                    #"generated_image.png")




if __name__ == '__main__':
    generate(50)
