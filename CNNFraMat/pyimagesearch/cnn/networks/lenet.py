# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras
"""
def conv_spec(c, h, w, input_shape, border_mode):
	return Convolution2D(c, h, w, border_mode="same", input_shape=input_shape)
"""

class LeNet:
	@staticmethod	
	def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()

		# first set of CONV => RELU => POOL
		model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(depth, height, width)))
		#model.add(conv_spec(20, 5, 5, (depth, height, width), "same"))
		model.add(Activation("relu"))

		# second set of CONV => RELU => POOL
		model.add(Convolution2D(50, 3, 3, border_mode="same"))
		model.add(Activation("relu"))

		# set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model