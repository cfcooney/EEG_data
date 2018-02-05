import numpy as np
import random
import pickle
import scipy.io as spio

def load_data():
    import os
    os.chdir("../data")

    data = spio.loadmat('Word_Data', squeeze_me=True) #Loading data from Matlab format file
    data = data['Data_Cell'] #Extracts data minus metadata
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
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) #retains the same 'random_state
   
    return X_train, X_test, y_train, y_test
	
data = load_data()
X_train, X_test, y_train, y_test= featuresets_and_labels(data)