# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : classification_NB.py
# Description : NB classifier for the youtube 8m dataset
# Author      : Fares Meghdouri

#******************************************************************************

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.naive_bayes import GaussianNB   
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

SEED = 2019
SCORING = ['accuracy', 'f1_weighted', 'jaccard_weighted']
CV = False

def read_data(input):

	X_train = np.load('data/training_data.npy')
	X_validate = np.load('data/validation_data.npy')
	y_train = np.load('data/training_labels.npy')
	y_validate = np.load('data/validation_labels.npy')

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

def tune(model, X, y, cv):

	min_samples_leaf_range = np.round(np.linspace(1, 10, 10)).astype(int)
	max_depth_range 	   = np.round(np.linspace(1, 30, 30)).astype(int)
	param_dist 			   = dict(min_samples_leaf=min_samples_leaf_range, max_depth=max_depth_range)
	num_features		   = len(X_train_little[0])

	best_model 			   = EvolutionaryAlgorithmSearchCV( estimator     	    = model,
															params              = param_dist,
															scoring             = "f1",
															cv                  = cv,
															verbose				= 1,
															population_size	    = 50,
															gene_mutation_prob  = 0.10,
															gene_crossover_prob = 0.5,
															tournament_size		= 3,
															generations_number	= 6,
															n_jobs				= 4)
	best_model.fit(X_train_little, y_train_little)

	return best_model

def cr():
	pass

def main():

	X_train, X_validate, y_train, y_validate = read_data('youtube8m_clean')
	NB = GaussianNB()
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	NB.fit(X_train, y_train)
	if CV:
		sc_tr = cross_validate(NB, X_train, y_train, scoring=SCORING, cv=cv, return_train_score=False)
		sc_ts = cross_validate(NB, X_validate, y_validate, scoring=SCORING, cv=cv, return_train_score=False)

		print("%0.3f (+/- %0.3f)" % (sc_tr['test_accuracy'].mean(), sc_tr['test_accuracy'].std() * 2))
		print("%0.3f (+/- %0.3f)" % (sc_ts['test_accuracy'].mean(), sc_ts['test_accuracy'].std() * 2))

		#print("%0.3f (+/- %0.3f)" % (sc_tr['test_f1_weighted'].mean(), sc_tr['test_f1_weighted'].std() * 2))
		#print("%0.3f (+/- %0.3f)" % (sc_ts['test_f1_weighted'].mean(), sc_ts['test_f1_weighted'].std() * 2))

		#print("%0.3f (+/- %0.3f)" % (sc_tr['test_jaccard_weighted'].mean(), sc_tr['test_jaccard_weighted'].std() * 2))
		#print("%0.3f (+/- %0.3f)" % (sc_ts['test_jaccard_weighted'].mean(), sc_ts['test_jaccard_weighted'].std() * 2))

	
	pred_validate = NB.predict(X_validate)
	pred_train = NB.predict(X_train)

	print('######## METRICS ############')
	print('######## TRAINING SET ############')
	print(accuracy_score(y_train, pred_train))
	print(f1_score(y_train, pred_train, average='weighted'))
	print(jaccard_score(y_train, pred_train, average='weighted'))
	print('######## VALIDATION SET ############')
	print(accuracy_score(y_validate, pred_validate))
	print(f1_score(y_validate, pred_validate, average='weighted'))
	print(jaccard_score(y_validate, pred_validate, average='weighted'))

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

	plot_confusion_matrix(y_train, pred_train, classes=np.unique(y_train), normalize=False,
                      title='NB training confusion matrix')
	plt.savefig('NB_tr_cm')

	plot_confusion_matrix(y_validate, pred_validate, classes=np.unique(y_train), normalize=False,
                      title='NB testing confusion matrix')
	plt.savefig('NB_ts_cm')

if __name__ == "__main__":
	main()