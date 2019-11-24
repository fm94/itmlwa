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
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

SEED = 2019
SCORING = ['accuracy', 'f1_weighted', 'jaccard_weighted']
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
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def main():

	X_train, X_validate, y_train, y_validate = read_data('youtube8m_clean')

	y_train_categorical = to_categorical(y_train,n_classes)
	y_validate_categorical = to_categorical(y_validate,n_classes)

	model = Sequential()
	model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
	model.add(Dense(48, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_classes, activation='sigmoid'))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=3, verbose=2, mode='auto')
	model.fit(X_train,y_train_categorical,batch_size=32,epochs=50,verbose=1,validation_split=0.20, callbacks=[earlyStopping])

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

	plot_confusion_matrix(y_train, pred_train, classes=np.unique(y_train), normalize=True,
                      title='ANN training confusion matrix')
	plt.savefig('ANN_tr_cm')

	plot_confusion_matrix(y_validate, pred_validate, classes=np.unique(y_train), normalize=True,
                      title='ANN testing confusion matrix')
	plt.savefig('ANN_ts_cm')

if __name__ == "__main__":
	main()
