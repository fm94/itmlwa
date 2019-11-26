# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : classification_LS.py
# Description : LS classifier for the youtube 8m dataset
# Author      : Fares Meghdouri

#******************************************************************************

from numpy.linalg import inv, solve, matrix_rank
import numpy as np
import sys, os
from random import shuffle, randint
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

SEED = 2019
chosen_labels = [19,23,33]
n_classes = len(chosen_labels)

def read_data(input):

	X_train = np.load('data/training_data.npy')
	X_validate = np.load('data/validation_data.npy')
	y_train = np.load('data/training_labels.npy')
	y_validate = np.load('data/validation_labels.npy')

	y_train = np.where(y_train==chosen_labels[0], 0, y_train)
	y_train = np.where(y_train==chosen_labels[1], 1, y_train)
	y_train = np.where(y_train==chosen_labels[2], 2, y_train)

	y_validate = np.where(y_validate==chosen_labels[0], 0, y_validate)
	y_validate = np.where(y_validate==chosen_labels[1], 1, y_validate)
	y_validate = np.where(y_validate==chosen_labels[2], 2, y_validate)

	return X_train, X_validate, y_train, y_validate

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=15)
    fig.tight_layout()
    return ax

def _train(x,y):
	"""
		Build the linear least weight vector W
		:param x: NxD matrix containing N attributes vectors for training
		:param y: NxK matrix containing N class vectors for training
	"""
	# D = Number of attributes
	D = x.shape[1] + 1
	# K = Number of classes
	K = y.shape[1]
	
	# Build the sums of xi*xi' and xi*yi'
	sum1 = np.zeros((D,D)) # init placeholder
	sum2 = np.zeros((D,K))
	i = 0
	for x_i in x:						# loop over all vectors
		x_i = np.append(1, x_i) 		# augment vector with a 1 
		y_i = y[i]						
		sum1 += np.outer(x_i, x_i)		# find xi*xi'
		sum2 += np.outer(x_i, y_i)		# find xi*yi'
		i += 1
	
	# Check that condition number is finite
	# and therefore sum1 is nonsingular (invertable)
	while matrix_rank(sum1) != D:
		# Naive choice of sigma.
		# Could cause inaccuracies when sum1 has small values
		# However, in most cases the matrix WILL be invertable
		sum1 = sum1 + 0.001 * np.eye(D) 
	
	# Return weight vector
	# Weight vector multiplies sums and inverse of sum1
	return np.dot(inv(sum1),sum2)


def predict(W, x):
	"""
	Predict the class y of a single set of attributes
	:param W:	DxK Least squares weight matrix
	:param x:	1xD matrix of attributes for testing
	:return:	List of 0's and 1's. Index with 1 is the class of x
	"""
	x = np.append(1, x)		# augment test vector

	# Solve W'*x
	values = list(np.dot(W.T,x))
	
	# Find maxima of values
	winners = [i for i, x in enumerate(values) if x == max(values)] # indexes of maxima
	# Flip a coin to decide winner
	# if only one winner, it will be chosen by default
	index = randint(0,len(winners)-1)
	winner = winners[index]

	y = [0 for x in values] 	# initalize list with all zeros
	y[winner] = 1 				# set winner
	return y

def fixLabels(y):
	return to_categorical(y,3)
	#return pd.get_dummies(y).values

def test(X_train, X_validate, y_train, y_validate):
	"""
	Runs the linear least squares classifier
	:param a:	All the data
	:param b:	All the classes corresponding to data
	:param split: 	Where to split data for training
					Ex: 40 trains with 40% and tests with 60%
	"""

	# Build weight vector from training data
	W = train(X_train, y_train)
	
	total = y_validate.shape[0]
	i = 0
	hits = 0
	# Predict the class of each xi, and compare with given class
	for i in range(total):
		prediction = predict(W,X_validate[i])
		actual = list(np.ravel(y_validate[i]))
		if prediction == actual:
			hits += 1
	accuracy = hits/float(total)*100
	print ("Accuracy = " + str(accuracy) + "%", "(" + str(hits) + "/" + str(total) + ")")

def classic_test(X_train, X_validate, y_train, y_validate):
	"""
	Runs the linear least squares classifier
	:param a:	All the data
	:param b:	All the classes corresponding to data
	:param split: 	Where to split data for training
					Ex: 40 trains with 40% and tests with 60%
	"""

	# Build weight vector from training data

	y_train = to_categorical(y_train,3)
	y_validate = to_categorical(y_validate,3)

	W = _train(X_train, y_train)


	total = y_train.shape[0]

	# Predict the class of each xi, and compare with given class
	prediction = []
	actual = []
	for i in range(total):
		prediction.append(predict(W,X_train[i]))
		actual.append(list(np.ravel(y_train[i])))

	# reverse dummies to 1d array
	prediction = [np.argmax(y, axis=None, out=None) for y in np.array(prediction)]
	actual = [np.argmax(y, axis=None, out=None) for y in np.array(actual)]

	print('######## METRICS ############')
	print('######## TRAIN SET ############')
	print(accuracy_score(actual, prediction))
	print(f1_score(actual, prediction, average='weighted'))
	print(jaccard_score(actual, prediction, average='weighted'))

	plot_confusion_matrix(actual, prediction, classes=np.unique(actual), normalize=False,
                      title='LS training confusion matrix')
	plt.savefig('LS_tr_cm')



	total = y_validate.shape[0]

	# Predict the class of each xi, and compare with given class
	prediction = []
	actual = []
	for i in range(total):
		prediction.append(predict(W,X_validate[i]))
		actual.append(list(np.ravel(y_validate[i])))

	# reverse dummies to 1d array
	prediction = [np.argmax(y, axis=None, out=None) for y in np.array(prediction)]
	actual = [np.argmax(y, axis=None, out=None) for y in np.array(actual)]

	print('######## METRICS ############')
	print('######## VALIDATION SET ############')
	print(accuracy_score(actual, prediction))
	print(f1_score(actual, prediction, average='weighted'))
	print(jaccard_score(actual, prediction, average='weighted'))

	plot_confusion_matrix(actual, prediction, classes=np.unique(actual), normalize=False,
                      title='LS testing confusion matrix')
	plt.savefig('LS_ts_cm')




def cv_test(X_train, X_validate, y_train, y_validate):
	"""
	Runs the linear least squares classifier
	:param a:	All the data
	:param b:	All the classes corresponding to data
	:param split: 	Where to split data for training
					Ex: 40 trains with 40% and tests with 60%
	"""

	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
	cvscores = []


	for train, test in kfold.split(X_train, y_train):
		# Build weight vector from training data

		y_train_c = to_categorical(y_train,3)
		y_validate_c = to_categorical(y_validate,3)

		print('training')
		x = X_train[train]
		y = y_train_c[train]
		W = _train(x, y)
	
		total = y.shape[0]

		# Predict the class of each xi, and compare with given class
		prediction = []
		actual = []
		print('testing')
		for i in range(total):
			prediction.append(predict(W,x[i]))
			actual.append(list(np.ravel(y[i])))

		# reverse dummies to 1d array
		prediction = [np.argmax(y, axis=None, out=None) for y in np.array(prediction)]
		actual = [np.argmax(y, axis=None, out=None) for y in np.array(actual)]

		cvscores.append(accuracy_score(actual, prediction)*100)
	print("accuracy over 5 folds for training set %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def main():

	X_train, X_validate, y_train, y_validate = read_data('youtube8m_clean')

	#cv_test(X_train, X_validate, y_train, y_validate)
	classic_test(X_train, X_validate, y_train, y_validate)

# Python doesnt call main() by default
if __name__ == "__main__":
	main()
