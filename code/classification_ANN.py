# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : classification_ANN.py
# Description : KNN classifier for the youtube 8m dataset
# Author      : Fares Meghdouri

#******************************************************************************

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

SEED = 2019
SCORING = ['f1_weighted']
tune = False
n_classes = 3

def read_data(input):

	X_train = np.load('data/training_data.npy')
	X_validate = np.load('data/validation_data.npy')
	y_train = np.load('data/training_labels.npy')
	y_validate = np.load('data/validation_labels.npy')

	y_train = np.where(y_train==15, 0, y_train)
	y_train = np.where(y_train==19, 1, y_train)
	y_train = np.where(y_train==21, 2, y_train)

	y_validate = np.where(y_validate==15, 0, y_validate)
	y_validate = np.where(y_validate==19, 1, y_validate)
	y_validate = np.where(y_validate==21, 2, y_validate)

	return X_train, X_validate, y_train, y_validate

def main():

	X_train, X_validate, y_train, y_validate = read_data('youtube8m_clean')

	y_train_categorical = to_categorical(y_train,n_classes)
	y_validate_categorical = to_categorical(y_validate,n_classes)

	model = Sequential()
	model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(n_classes, activation='sigmoid'))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto')
	model.fit(X_train,y_train_categorical,batch_size=32,epochs=50,verbose=1,validation_split=0.05, callbacks=[earlyStopping])

	# evaluate the keras model
	#_, accuracy = model.evaluate(X_validate, y_validate_categorical)
	#print(accuracy)

	pred_validate = model.predict_classes(X_validate)
	pred_train = model.predict_classes(X_train)

	print('######## C-MATRIX ############')

	print('######## TRAINING SET ############')
	print(confusion_matrix(y_train, pred_train), '\n\n')
	print('######## VALIDATION SET ############')
	print(confusion_matrix(y_validate, pred_validate), '\n\n')

	print('######## REPORT ############')

	print('######## TRAINING SET ############')
	print(classification_report(y_train, pred_train), '\n\n')
	print('######## VALIDATION SET ############')
	print(classification_report(y_validate, pred_validate), '\n\n')

if __name__ == "__main__":
	main()