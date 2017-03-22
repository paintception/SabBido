import numpy as np
import argparse
import cv2
import seaborn
import keras

from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

def load_dataset():
	return(load_digits())

def shape_data(dataset):
	
	data = dataset.data.reshape((dataset.data.shape[0], 8, 8))
	shaped_data = data[:, np.newaxis, :, :]

	return shaped_data

def make_categorical(labels, n_classes):
	return(np_utils.to_categorical(labels, n_classes))

def plots(history):

	f1 = plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	plt.show()

	f2 = plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	plt.show()

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

	ap = argparse.ArgumentParser()
	ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
	ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
	ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
	args = vars(ap.parse_args())

	data = load_dataset()
	dataset = shape_data(data)

	trainData, testData, trainLabels, testLabels = train_test_split(dataset / 255.0, data.target.astype("int"), test_size=0.10)

	tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/matthia/Desktop/ogs', histogram_freq=0, write_graph=True, write_images=False)

	trainLabels = make_categorical(trainLabels, 10)
	testLabels = make_categorical(testLabels, 10)

	print("[INFO] compiling model...")
	opt = SGD(lr=0.01)

	model = LeNet.build(width=8, height=8, depth=1, classes=10, weightsPath=args["weights"] if args["load_model"] > 0 else None)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	history = model.fit(trainData, trainLabels, batch_size=50, nb_epoch=40, verbose=1, validation_data=(testData, testLabels),callbacks=[tbCallBack])

	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

	plots(history)

if __name__ == '__main__':
	main()

