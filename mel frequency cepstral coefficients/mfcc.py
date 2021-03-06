"""
Name: Ciaran Cooney
Date: 09/05/2018
Suite of functions for computing mel frequency cepstral coefficients step-by-step
"""
import numpy as np 
import math 
from scipy.fftpack import dct

#####Step One#####
def segment_data(data,window,overlap):
	data_frames = []
	window_len = int(len(data[1,:])*window)
	window_ovlp = int(len(data[1,:])*overlap)
	n_windows = int((len(data[1,:])/window_len)*2 - 1)
	n_channels = len(data[:,1])

	start = 0
	end = window_len 

	for i in range(n_windows):
            data_frames.append(data[0:n_channels,start:end])
            start += window_ovlp
            end += window_ovlp      
	return np.array(data_frames)

#####Step Two#####
def periodogram(data_frame, fs=1000, nfft=512):
	power_spectrum = []
	for frame in data_frame:
		pk_chan = []
		for chan in frame:
			sk = np.absolute(np.fft.fft(chan,nfft))
			pk = 1/nfft*np.square(sk) 
			pk_chan.append(pk)
		power_spectrum.append(pk_chan)
	return power_spectrum


#####Step Three#####
def mel_filterbank(n_filterbanks, freq_low, freq_high, nfft,fs):
	mels = []
	mels_hz = []
	fft_bin_num = []
	freq_range = [freq_low,freq_high]
	for f in freq_range:
		mf = 1125*math.log(1+f/700)
		mels.append(mf)

	mi = np.linspace(mels[0],mels[1],num=n_filterbanks)

	for m in mi:
		m_i = 700*(np.exp(m/1125)-1)
		mels_hz.append(m_i)

	for i in mels_hz:
		fft_bin= math.floor((nfft+1)*i/fs)
		fft_bin_num.append(fft_bin)
	return fft_bin_num

#####Step Four#####
def get_filterbanks(n_filterbanks=20, nfft=512, fs=1000, freq_low=0, freq_high=500):
	bin = mel_filterbank(n_filterbanks,freq_low,freq_high,nfft,fs)
	fbank = np.zeros([n_filterbanks,nfft])
	for i in range(1, n_filterbanks - 1):
		minus_1 = int(bin[i - 1])
		centre = int(bin[i])
		plus_1 = int(bin[i + 1])

		for j in range(minus_1, centre):
			fbank[i-1, j] = (j - bin[i - 1]) / (bin[i] - bin[i - 1])
		for j in range(centre, plus_1):
			fbank[i-1, j] = (bin[i + 1] - j) / (bin[i + 1] - bin[i])	
	return fbank

#####Step Five#####
def log_filterbank_energies(pow_spectrum, fbank):
	feat = np.dot(pow_spectrum, fbank.T)
	feat = np.where(feat == 0, np.finfo(float).eps,feat)

	fb_energies = np.log(feat) # possible change -- 20 * np.log10(filter_banks)
	return fb_energies

#####Step Six#####
def discrete_cosine_transform(fb_energies, n_cep):
	coeffs = dct(fb_energies, type=2, norm='ortho')[:,1:(n_cep + 1)]
	return coeffs

"""
#Test Function
import scipy.io as spio
path = "C:/Users\cfcoo\OneDrive - Ulster University\KaraOne\Data\MM09"
def load_data(path, file, column):
    import os
    os.chdir(path)

    data = spio.loadmat (file, squeeze_me=True) #Loading data from Matlab format file
    data = data[column] #Extracts data minus metadata
    return data
import pandas as pd
data = load_data(path, 'EEG_Data','EEG_Data')
data = data['EEG']
data = np.ravel(data) 
data = pd.DataFrame(data[0])
data = data[0]
data = data[0]

windows = segment_data(data,0.1,0.05)
pow_spectrum = periodogram(windows, fs=1000, nfft=512)
fbank = get_filterbanks()
fb_energies = log_filterbank_energies(pow_spectrum=pow_spectrum, fbank=fbank)
coeffs = discrete_cosine_transform(fb_energies, 12)
print(coeffs.shape)
"""
