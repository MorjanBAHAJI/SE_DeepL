import os
import scipy
import numpy as np
import librosa
from scipy.io.wavfile import read
from utils.path import path_n_to_any


def get_ori(sound_path):
    sound_path_n = sound_path[0]
    sound_path_no = sound_path[1]
    _, y_n = read(sound_path_n)
    _, y_no = read(sound_path_no)
    return (y_n - y_no) 

def get_spectro(y, dB=True):
    _,_,z = scipy.signal.stft(y, fs=16000, nperseg=256)     
    mag , phase = librosa.core.magphase(z)
    if dB:
        mag = librosa.core.amplitude_to_db(mag)
    return mag,phase

def sound_to_images(sound_path, dB=True):
    if isinstance(sound_path, (list)):
        y = get_ori(sound_path)
    else:    
        _, y = read(sound_path)
        
    mag , phase = get_spectro(y, dB)
    return phase, mag[..., np.newaxis]

def images_to_sound(phase, mag, path_to_save=None):                                               
    zfinal = mag*phase
    _,file = scipy.signal.istft(zfinal)
    #file = np.around(file, decimals=0).astype(np.int16)
    if path_to_save:
        scipy.io.wavfile.write(path_to_save,16000,file)
    return file

def mask(path_n, path_no, type_mask):
    
    _,mag=sound_to_images([path_n,path_no], dB=False)
    mag = mag.astype(np.float64)
    
    _,mag_b=sound_to_images(path_no, dB=False)
    mag_b = mag_b.astype(np.float64)
    
    if (type_mask=='Binary'):
        mask_binary=np.square(mag)-np.square(mag_b) 
        mask_binary=mask_binary>0
        mask_binary=mask_binary.astype(int)
        return mask_binary
    else :
        clean_matrice_square=mag*mag
        bruit_matrice_square=mag_b*mag_b
        numerator=clean_matrice_square
        denominator=clean_matrice_square+bruit_matrice_square
        mask_soft=np.true_divide(numerator,denominator)
        return mask_soft
     
def n_to_any(l_names_noisy,path_no,train_test='train',masktype=None,output='no'):
    maxmin=[73.94335, -49.70791]
    max_all=maxmin[0]
    min_all=maxmin[1]
    X = []
    Y = []
    P_n = [] #Phase
    P_ori = []
    if output=='ori':
        for name_n in l_names_noisy:
            _, name_no = path_n_to_any(name_n,path_no)
            phase_n, mag_n = sound_to_images(name_n, dB=True)
            phase_ori, mag_ori = sound_to_images([name_n,name_no], dB=True)
            X.append((mag_n-min_all)/(max_all-min_all))
            Y.append((mag_ori-min_all)/(max_all-min_all))
            P_ori.append(phase_ori)
            P_n.append(phase_n)
    else:
        for name_n in l_names_noisy:
            if masktype!='Binary' and masktype!='Soft' :
                raise ValueError("Please choose mask type: Binary or Soft")
                
            _,name_no = path_n_to_any(name_n,path_no)
            phase_n, mag_n = sound_to_images(name_n, dB=True)
            phase_ori, _ = sound_to_images([name_n,name_no], dB=True)
         
            X.append((mag_n-min_all)/(max_all-min_all))
            Y.append(mask(name_n, name_no,masktype))
            P_ori.append(phase_ori)
            P_n.append(phase_n)
    return np.array(X), np.array(Y), np.array(P_ori), np.array(P_n) 



                         
            
            
            
            
            
 
            
            
            
            
            
            
            
