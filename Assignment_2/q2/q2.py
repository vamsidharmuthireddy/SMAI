import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
import argparse
import pdb
import pandas as pd
import numpy as np
import theano
from sklearn import svm

# theano.config.device = 'gpu'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
os.environ['THEANO_FLAGS'] = 'device=gpu'
seed = 7
numpy.random.seed(seed)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help="Path to the data")
ap.add_argument("-o", "--model_name", required=True, help="saved model")
ap.add_argument("-a", "--activation", default='relu', help="activation funtion to be used")
ap.add_argument("-m", "--model", default='Model_baseline', help="model to be used")
ap.add_argument("-e", "--epochs", type=float, default=2, help="no. of epochs")
ap.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate")
ap.add_argument('--b', action='store_true', default=True)
args = vars(ap.parse_args())

def unpickle(file):
    import pickle

    with open(file, 'rb') as fo:
        mydict = pickle.load(fo)
    return mydict

def load_data(path):
	# data = np.zeros((50000, 3072))
	data = np.zeros((20000, 3072))
	# labels = np.zeros((50000), dtype=int)
	labels = np.zeros((20000), dtype=int)
	for i in range(1,3):
		data_dict = unpickle(os.path.join(path,'data_batch_%d'%i))
		row = (i-1)*10000
		data[row:row+10000, :] = data_dict['data']
		labels[row:row+10000] = (data_dict['labels'])
	return {'data':data, 'labels': labels} 

def split_data(data_dict):
	train_data = {'data': data_dict['data'][:int(0.8*len(data_dict['data']))], 
		'labels': data_dict['labels'][:int(0.8*len(data_dict['labels']))]}
	test_data = {'data': data_dict['data'][int(0.8*len(data_dict['data'])):], 
		'labels': data_dict['labels'][int(0.8*len(data_dict['labels'])):]}
	return train_data, test_data

def get_accuracy(predictions, y_test):
	
	acc = sum((list(map(lambda x, y: 1 if x==y else 0,predictions,y_test))))/float(len(predictions))
	return acc
def Model_baseline():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32)))
	model.add(Activation(args['activation']))
	
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation(args['activation']))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation(args['activation']))
	model.add(Dense(num_classes, activation='softmax'))
	return model



def Model_dropout():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	

	model.add(Conv2D(32, (3, 3), padding='same', kernel_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	return model

def Model_batch_norm():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation(args['activation']))
	
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation(args['activation']))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation(args['activation']))
	model.add(Dense(num_classes, activation='softmax'))
	return model

def Model_deep():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same'))	
	model.add(Activation(args['activation']))

	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), padding='same'))	
	model.add(Activation(args['activation']))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), padding='same'))	
	model.add(Activation(args['activation']))
	#model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation(args['activation']))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model


	
if "__name__" != "__main__":

	path = args['filename']
	all_data = load_data(path)
	train, test = split_data(all_data)
	
	X_train = train['data'].astype('float32')
	X_test = test['data'].astype('float32')
	X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
	X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

	X_train = X_train / 255.0
	X_test = X_test / 255.0
	y_train = np_utils.to_categorical(train['labels'])
	y_test = np_utils.to_categorical(test['labels'])
	num_classes = y_test.shape[1]
	if args['model'] == 'Model_baseline':
		model = Model_baseline()
	elif args['model'] == 'Model_dropout':
		model = Model_dropout()
	elif args['model'] == 'Model_batch_norm':
		model = Model_batch_norm()
	elif args['model'] == 'Model_deep':
		model = Model_deep()
	elif args['model'] == 'Model':
		model = Model()


	filepath = args['model_name']
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	epochs = args['epochs']
	lrate = args['learning_rate']
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args['epochs'], batch_size=32, callbacks=callbacks_list,verbose=0)

	# Final evaluation of the model
	if args['b']:
		scores = model.evaluate(X_test, y_test, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))
	else:
		features = model.predict(X_test)
		clf = svm.SVC()
		clf.fit(features, y_train)
		predicted= clf.predict(X_test)
		predictions = [round(x) for x in predicted]
		accuracy = get_accuracy(predictions)
		print("Accuracy for SVM: %.2f%%" % (accuracy*100))