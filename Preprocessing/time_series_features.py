"""
Name: Ciaran Cooney
Date:18/03/2018
Script for loading windowed EEG data and returning
feature vectors for each trial contianing time-series
features.
"""

import numpy as np
import pickle 
from import_data import load_pickle
from features import simple_features, feature_array 
folders = ['MM09', 'MM10', 'MM11', 'MM12', 'MM14','MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21']
path = "C:/Users\SB00745777\OneDrive - Ulster University\KaraOne\Data/"

for f in folders:
    new_path = path + f
    data = load_pickle(new_path,"window_data.p")
    labels = load_data(new_path,"labels","labels")
    feature_vector = []
    for tr in data:
        
        wdw_mn,wdw_absmn,wdw_sm,wdw_sd,wdw_md,wdw_vr,wdw_mx,wdw_absmx,wdw_mnm,wdw_absmin,wdw_mxmn,wdw_mnmx = ([] for i in range(12))
        for wdw in tr:
            ch_mn,ch_absmn,ch_sm,ch_sd,ch_md,ch_vr,ch_mx,ch_absmx,ch_mnm,ch_absmin,ch_mxmn,ch_mnmx = ([] for i in range(12))
            for ch in wdw:
                mn,absmn,sd,sm,md,vr,mx,absmx,mnm,absmin,mxmn,mnmx = simple_features(ch)
                ch_mn.append(mn)
                ch_absmn.append(absmn)
                ch_sm.append(sm)
                ch_sd.append(sd)
                ch_md.append(md)
                ch_vr.append(vr)
                ch_mx.append(mx)
                ch_absmx.append(absmx)
                ch_mnm.append(mnm)
                ch_absmin.append(absmin)
                ch_mxmn.append(mxmn)
                ch_mnmx.append(mnmx)
            wdw_mn.append(ch_mn)
            wdw_absmn.append(ch_absmn)
            wdw_sm.append(ch_sm)
            wdw_sd.append(ch_sd)
            wdw_md.append(ch_md)
            wdw_vr.append(ch_vr)
            wdw_mx.append(ch_mx)
            wdw_absmx.append(ch_absmx)
            wdw_mnm.append(ch_mnm)
            wdw_absmin.append(ch_absmin)
            wdw_mxmn.append(ch_mxmn)
            wdw_mnmx.append(ch_mnmx)

            all_features = [wdw_mn,wdw_absmn,wdw_sm,wdw_sd,wdw_md,wdw_vr,wdw_mx,wdw_absmx,wdw_mnm,wdw_absmin,wdw_mxmn,wdw_mnmx]
            trial = np.array(())
            for window in all_features:
                data = feature_array(window)
                trial = np.append(trial,data)
       
        feature_vector.append(trial)
    
    df = {'Features':feature_vector,'Targets':labels}
    df = pd.DataFrame(df)
    pickle.dump(df, open("td_df.p", "wb")) #features and targets together in DataFrame
    pickle.dump(feature_vector, open("td_features.p", "wb")) #features only        
                    
        			
        			
        				
    		       	
        			
    				
    				
    				
    				
    				
    						
    			
    			
    			
    			
    			
    			
    			
    			
    			
    			
    			
    			
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
    		
				
