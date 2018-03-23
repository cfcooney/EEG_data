"""
Name: Ciaran Cooney
Date:10/03/2018
Script for importing features and targets, splitting the data, 
performing PCA and training a Decision Tree classifier.
"""

import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, KFold, cross_val_predict 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users\SB00745777\OneDrive - Ulster University\KaraOne\Data Preprocessing')
from import_data import load_pickle
from process_functions import order_class_labels

folders = ['MM09', 'MM10', 'MM11', 'MM12', 'MM14','MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21'] #folder names
path = "C:/Users\SB00745777\OneDrive - Ulster University\KaraOne\Data/"
classes = ['/uw/','/tiy/','/iy/','/m/','/n/','/piy/','/diy/','gnaw','pat','knew','pot'] #class labels

classifier_scores = pd.DataFrame() #DataFrame for saving CV scores

#####Iterate through subject folders and train classifier#####
for f in folders:
	new_path = path + f 
	data = load_pickle(new_path,"td_df.p") 
	features = data.Features.tolist()
	targets = np.array(data.Targets)

	class_names = order_class_labels(classes,targets) #orders class_names in the order they first appear in dataset
	X_Train, X_Test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
	
	clf = DecisionTreeClassifier(criterion="gini")
	pca = PCA(whiten=False)

	pipeline = Pipeline(steps=[('pca', pca), ('clf', clf)])

	pca.fit(X_Train)

	#####Plot figure for variance ratio#####
	fig = plt.figure(1, figsize=(10, 8))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')

	#####Grid Search on parameters#####
	n_components = [5, 10, 20, 30, 40, 50]
	max_leaf_nodes = [50, 100, 200, 300, 400, 500]

	estimator = GridSearchCV(pipeline,
	                         dict(pca__n_components=n_components,
	                              clf__max_leaf_nodes=max_leaf_nodes))

	#####initialise cross-validation and make predictions#####
	cv = KFold(len(y_train), 5, shuffle=True, random_state=42)
	scores = cross_val_score(estimator, X_Train, y_train, cv=cv, n_jobs=1)
	pred = cross_val_predict(estimator, X_Train, y_train, cv=cv, n_jobs=1)

	print("Classification Accuracy for subject " + f + ": " + str(np.mean(scores)))

	estimator.fit(features, targets)
	print(estimator.best_estimator_)

	plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
	            linestyle=':', label='n_components chosen')
	plt.legend(prop=dict(size=12))
	fig.savefig('pca.png') #save figure to subject folder

	sys.path.append("../Data Preprocessing")
	from visualisation import plot_confusion_matrix 
	cnf_matrix = confusion_matrix(y_train, pred)
	#print(cnf_matrix) 

	fig_1 = plt.figure(figsize=(10, 8))#initialise figure for confusion matrix
	
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                      title='Confusion matrix')

	fig_1.savefig('conf_mat.png')

	pred_ho = estimator.predict(X_Test)
	accuracy = accuracy_score(pred_ho,y_test)
	print("Classifier Accuracy (hold-out data) for subject " + f + ": " + str(accuracy))

	classifier_scores[f] = scores #add CV scores to dataframe
	plt.show(block=False)

#####CV scores saved as csv#####
classifier_scores.to_csv('scores.csv')

	
   
	
	