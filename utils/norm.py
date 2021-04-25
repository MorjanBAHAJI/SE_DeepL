import librosa
from utils.data_transfo import images_to_sound
from utils.path import path_n_to_any
import numpy as np

def norm_to_N(tab_in,N=1):
    Pav = sum(abs(tab_in)**2)
    norm  = 1/np.sqrt(Pav/N)
    tab_norm = tab_in*norm
    Pap = sum(abs(tab_norm)**2)
    return tab_norm


def unnorm(mag):
    mag = (mag*(73.94335+49.70791))-49.70791
    return librosa.core.db_to_amplitude(mag)

def unorm_to_audio(X,Y,P_ori,P_n,output,masktype,l_noisy,net,SNR,save='all'):
    X_in=[]
    Y_out=[]
    file_name = []
    [file_name.append(path_n_to_any(name_n)) for name_n in l_noisy]

    #Define output saving path
    if masktype:
        output = output +'_'+ masktype + '_'
    if save=='all':
        path_x = os.path.join('audio_test','in_'+SNR+'_'+file_name[i])
        path_y = os.path.join('audio_test','out_'+net+'_'+SNR+'_'+output+file_name[i])
    elif save=='only_y':
        path_x = None
        path_y = os.path.join('audio_test','out_'+net+'_'+SNR+'_'+output+file_name[i])
    elif save=='none':
        path_x = None
        path_y = None
    
    #Unnorm and reconstruction based on the output
    if output=='ori':
        for i,name in enumerate(l_noisy):
            x=unnorm(X[i,:,:,0])
            y=unnorm(Y[i,:,:,0])
     
            X_in.append(images_to_sound(P_n[i,:,:], x, 
                                         path_to_save=path_x))
            Y_out.append(images_to_sound(P_ori[i,:,:], y, 
                                         path_to_save=path_y))           
    else:
        for i,name in enumerate(l_noisy):
            x=unnorm(X[i,:,:,0])
            y=Y[i,:,:,0]
            X_in.append(images_to_sound(P_n[i,:,:], x, 
                                         path_to_save=path_x))
            Y_out.append(images_to_sound(P_ori[i,:,:], x*y, 
                                         path_to_save=path_y))
            
    return X_in,Y_out