# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : classification_NB.py
# Description : preprocess the youtube 8m dataset
# Author      : Fares Meghdouri

#******************************************************************************

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

SEED = 2019
SCORING = ['f1_weighted']
tune = False

def read_data(input):

	X_train = np.load('data/training_data.npy')
	X_validate = np.load('data/validation_data.npy')
	y_train = np.load('data/training_labels.npy')
	y_validate = np.load('data/validation_labels.npy')

	return X_train, X_validate, y_train, y_validate

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
	RF = RandomForestClassifier()
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
	if tune:
		best_model = tune(RF, X_train, y_train, cv)
		RandomForestClassifier(min_samples_leaf = rnds.best_estimator_.min_samples_leaf, 
							   max_depth        = rnds.best_estimator_.max_depth,
							   random_state     = SEED)
	RF.fit(X_train, y_train)
	sc_tr = cross_validate(RF, X_train, y_train, scoring=SCORING, cv=cv, return_train_score=False)
	sc_ts = cross_validate(RF, X_validate, y_validate, scoring=SCORING, cv=cv, return_train_score=False)

	print("%0.3f (+/- %0.3f)" % (sc_tr['test_f1_weighted'].mean(), sc_tr['test_f1_weighted'].std() * 2))
	print("%0.3f (+/- %0.3f)" % (sc_ts['test_f1_weighted'].mean(), sc_ts['test_f1_weighted'].std() * 2))
	
	pred_validate = RF.predict(X_validate)
	pred_train = RF.predict(X_train)

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