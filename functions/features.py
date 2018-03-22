"""
Name: Ciaran Cooney
Date:18/02/2018
Functions for facilitating feature extraction 
from EEG dataset.
"""
import numpy as np

def simple_features(chan):
    mn = np.mean(chan)   
    absmn = np.abs(mn)   
    sd = np.std(chan)    
    sm = np.sum(chan)    
    md = np.median(chan) 
    vr = np.var(chan)    
    mx = np.amax(chan)   
    absmx = np.abs(mx)  
    mnm = np.amin(chan)  
    absmin = np.abs(mnm) 
    mxmn = mx + mnm      
    mnmx = mx-mnm        
    
    return mn,absmn,sd,sm,md,vr,mx,absmx,mnm,absmin,mxmn,mnmx

#####Convert list data into numpy array#####
def feature_array(window):
    d = np.array(())
    for f in window:
        f = np.array(f)
        d = np.append(d,f)
    return d