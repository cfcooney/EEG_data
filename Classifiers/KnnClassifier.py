"""
Name: Ciaran Cooney
Date:10/03/2018
Script for importing features and targets, splitting the data, 
performing PCA and training a K-Nearest Neighbour classifier.
"""

import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score 
import scipy.io as spio 
from sklearn.model_selection import train_test_split
from visualisation import plot_confusion_matrix
import matplotlib.pyplot as plt
from process_functions import order_class_labels

data = spio.loadmat("format_features", squeeze_me=True)
features = data["all_trial_features"]
labels = spio.loadmat("labels", squeeze_me=True)
labels = labels["labels"]
feature_names = spio.loadmat("feature_names",squeeze_me=True)
feature_names = feature_names["feature_names"]

classes = ['/uw/','/tiy/','/iy/','/m/','/n/','/piy/','/diy/','gnaw','pat','knew','pot']
class_names = order_class_labels(classes,targets) #orders class_names in the order they first appear in dataset

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

n_components = 14
pca = PCA(n_components=n_components, whiten=False)

pca.fit(features)
print(pca.explained_variance_ratio_)

features_train_pca = pca.transform(X_train) 
features_test_pca = pca.transform(X_test)

#####Fit KNN Classifier#####
from sklearn.neighbors import KNeighborsClassifier 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features_train_pca, y_train) 

pred = clf.predict(features_test_pca)
accuracy = accuracy_score(pred, y_test)
print("Classifier Accuracy: " + str(accuracy))

#####Compute and plot confusion matrix#####
cnf_matrix = confusion_matrix(y_test, pred)
print(cnf_matrix) 

plt.figure(figsize = (8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix')