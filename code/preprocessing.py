# -*- coding: utf-8 -*-

#******************************************************************************
#
# Name        : preprocessing.py
# Description : preprocess the youtube 8m dataset
# Author      : Fares Meghdouri

#******************************************************************************

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

#******************************************************************************
SEED = 2019
use_pca = True
use_feature_selection = True
pca_component = 3

chosen_labels = ['19','23','33']

input_train = '/localdisk/fm-youtube8m/train_full_{}.csv'.format('_'.join(chosen_labels))
input_validation = '/localdisk/fm-youtube8m/validate_full_{}.csv'.format('_'.join(chosen_labels))
#******************************************************************************

def read_data(input):
	""" Read the data into pandas dataframes """

	print('>> Reading Data...')

	training = pd.read_csv(input_train).fillna(0)
	validation = pd.read_csv(input_validation).fillna(0)

	features = validation.columns

	X_train = training.drop(['label'], 1)
	X_validation = validation.drop(['label'], 1)

	# TODO: drop id also

	y_train = training['label']
	y_validation = validation['label']

	del training
	del validation

	print('>> Data Read. Found {} data samples and {} features'.format(X_train.shape[0], X_train.shape[1]))
	return X_train, X_validation, y_train, y_validation, features

def min_max_scaling(X_train, X_validation):
	""" Min-Max scaling of the data """

	print('>> Min-Max Scaling...')

	scaler = MinMaxScaler()
	scaler.fit(X_train)
	return scaler.transform(X_train), scaler.transform(X_validation)

def std_scaling(X_train, X_validation):
	""" Min-Max scaling of the data """

	print('>> Standard Scaling...')

	scaler = StandardScaler()
	scaler.fit(X_train)
	return scaler.transform(X_train), scaler.transform(X_validation)

def explain_PCA(X_train, X_validation, features):
	""" Perform PCA and plot space variances """

	print('>> PCA Analysis...')
	global pca_component

	covar_matrix = PCA(n_components = pca_component)
	#covar_matrix = PCA()
	covar_matrix.fit(X_train)
	variance = covar_matrix.explained_variance_ratio_
	#var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
	var=np.cumsum(covar_matrix.explained_variance_ratio_*100)

	plt.plot(var, lw=3)
	plt.grid()
	plt.ylabel('% Cumulative Variance', fontsize=20)
	plt.xlabel('# of Features', fontsize=20)
	plt.title('PCA Analysis', fontsize=20)
	plt.savefig('PCA_Analysis')
	print('>> PCA analysis results done, plot saved at: {}'.format('PCA_Analysis.png'))
	X_train_pca = covar_matrix.transform(X_train)
	X_validation_pca = covar_matrix.transform(X_validation)
	print('>>>> reduction {} to {}'.format(X_train.shape, X_train_pca.shape))
	print('>>>> reduction {} to {}'.format(X_validation.shape, X_validation_pca.shape))
	return X_train_pca, X_validation_pca
	#return X_train, X_validation

def feature_selection(X_train_pca, y_train, X_validation_pca):
	""" Feature selection based on Decision Trees feature importance """

	print('>> Feature selection based on DT importance...')

	dtree = DecisionTreeClassifier(random_state=SEED)
	dtree.fit(X_train_pca, y_train)
	# drop features with 0 importance
	indices = np.where(dtree.feature_importances_ == 0)
	X_train = np.delete(X_train_pca, indices[0], axis=1)
	X_validation  = np.delete(X_validation_pca, indices[0], axis=1)

	print('>> {} features were removed'.format(len(indices[0])))
	return X_train, X_validation

def main():

	X_train, X_validation, y_train, y_validation, features = read_data('youtube8m')
	X_train, X_validation = min_max_scaling(X_train, X_validation)
	X_train, X_validation = std_scaling(X_train, X_validation)
	if use_pca:
		X_train_pca, X_validation_pca = explain_PCA(X_train, X_validation, features)
	else:
		X_train_pca, X_validation_pca = X_train, X_validation
	if use_feature_selection:
		X_train_clean, X_validation_clean = feature_selection(X_train_pca, y_train, X_validation_pca)
	else:
		X_train_clean, X_validation_clean = X_train_pca, X_validation_pca

	X_train_clean, X_validation_clean = std_scaling(X_train_clean, X_validation_clean)
	
	os.mkdir('data')
	np.save('data/training_data.npy', X_train_clean)
	np.save('data/validation_data.npy', X_validation_clean)
	np.save('data/training_labels.npy', y_train.values)
	np.save('data/validation_labels.npy', y_validation.values)

if __name__ == "__main__":
	main()
