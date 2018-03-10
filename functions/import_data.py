"""
Name: Ciaran Cooney
Date: 24/02/2018
Script to import windowed data and compute linear features 
"""

import numpy as np 
import scipy.io as spio 

def load_data(path, file, column):
    import os
    os.chdir(path)

    data = spio.loadmat (file, squeeze_me=True) #Loading data from Matlab format file
    data = data[column] #Extracts data minus metadata
    return data

# Function to split features and labels
def targetFeatureSplit(data):
    target = []
    features = []
    for item in data:
        features.append( item[0] )
        target.append( item[1:] )

    return target, features

def featuresets_and_labels(data):
    from sklearn.model_selection import train_test_split
    data = np.nan_to_num(data) #Removes 'NaN' values - sets to zero
    
    target,features = targetFeatureSplit(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)#retains the same 'random_state
   
    return X_train, X_test, y_train, y_test