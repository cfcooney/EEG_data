#Functions for normalising feature values
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = []
def featureScaling(arr):
    Xmax = np.amax(arr)
    Xmin = np.amin(arr)
    for i in arr:
        x = float(i - Xmin)/(Xmax - Xmin)
        X.append(x)
    return X

def min_max_scal(x,feature_range):
	feature_range = feature_range
	scaler = MinMaxScaler(feature_range = feature_range)
	scaler.fit(x)
	return scaler.transform(x)