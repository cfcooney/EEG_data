"""
Name: Ciaran Cooney
Date:18/02/2018
Simple function for computing simple features
including mean, variance and standard deviation.
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