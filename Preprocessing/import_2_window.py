"""
Name: Ciaran Cooney
Date: 10/03/2018
Script for importing KaraOne EEG data in Matlab format
and converting it to 500ms windows
"""


import numpy as np
import pickle
#####Load Matalb file and format#####
from import_data import load_data 

folders = ['P02', 'MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14','MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21']
path = "C:/Users\SB00745777\OneDrive - Ulster University\KaraOne\Data/" 

#####Variables required for computing windows#####
samples = 5000
window_size = .1
window_samples = samples*window_size
window_ratio = samples/window_samples
n_windows = int(window_ratio*2 - 1)

#####Open folders/files, window and save#####
for f in folders:
    print("Computing windows for folder " + f)
    new_path = path + f
    data = load_data(new_path,"EEG_Data","EEG_Data")
    data = data['EEG']
    data = np.ravel(data) 
  
    window_all_trials = []
    all_windows = []
    for tr in data:
        tr_tp = np.transpose(tr)
        n_bins = int(np.round(len(tr_tp)/window_ratio))
        ovlp = float(n_bins/2) 
        window = [] 
        window.append(tr[0:62,0:n_bins])
        
        start = int(np.round(ovlp+1))
        end = int(np.round(n_bins + ovlp+2))
        m = 1

        for i in range(0, n_windows - 2):
            window.append(tr[0:62,start:end])
            start = (ovlp*(m+1)+1)
            start = int(np.round(start))
            end = int(np.round(end + ovlp))
            i += 1
            m += 1

        window.append(tr[0:62,start:])  
        all_windows.append(window)

    pickle.dump(all_windows, open("window_data.p", "wb"))
