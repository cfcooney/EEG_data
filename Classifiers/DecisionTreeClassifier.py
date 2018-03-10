"""
Name: Ciaran Cooney
Date:10/03/2018
Script for importing features and targets, splitting the data, 
performing PCA and training a Decision Tree classifier.
"""


import os
import scipy.io as spio 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
from visualisation import plot_confusion_matrix
import matplotlib.pyplot as plt
from process_functions import order_class_labels

features = spio.loadmat("features",squeeze_me=True)
features = features["features"]
targets = spio.loadmat("targets",squeeze_me=True)
targets = targets["target"]
feature_names = spio.loadmat("feature_names",squeeze_me=True)
feature_names = feature_names["feature_names"]

classes = ['/uw/','/tiy/','/iy/','/m/','/n/','/piy/','/diy/','gnaw','pat','knew','pot']
class_names = order_class_labels(classes,targets) #orders class_names in the order they first appear in dataset

X_Train, X_Test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

#####Principal Component Analysis#####
n_components = 14 
pca = PCA(n_components=n_components, whiten=False)

pca.fit(features)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

X_Train_pca = pca.transform(X_Train) 
X_Test_pca = pca.transform(X_Test)

#####Fit Decision Tree Classifier#####
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=300)
print("Fitting Decision Tree Classifier...")
clf.fit(X_Train_pca, y_train)

pred = clf.predict(X_Test_pca)
accuracy = accuracy_score(pred, y_test)
print("Classifier Accuracy: " + str(accuracy))

#####Compute and plot confusion matrix#####
cnf_matrix = confusion_matrix(y_test, pred)
print(cnf_matrix) 

plt.figure(figsize = (8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix')
