import os
import warnings
import numpy as np
from random import randrange
from tqdm import tqdm
import scipy
from scipy.io.wavfile import read
from utils.norm import norm_to_N

def random_part_noise(noise_path,len_y, parts=2):
    fs_noise, y_noise = read(noise_path)
    N=len_y
    limit=(len(y_noise)//N)+1
    noise=[]
    for i in range(parts):
        i=randrange(1,limit)
        noise.append(y_noise[(i-1)*N:i*N])
    noise_final=sum(noise)
    noise_final = noise_final.astype(np.float32)
    noise_norm = norm_to_N(noise_final,1)
    return noise_norm
    
def normalisation_and_sum_noise(signal,noise_final,SNR=0):
    Ps =sum(signal**2)
    Pb = sum(noise_final**2)
    ratio = Ps/(10**(SNR/10))
    noise_final = norm_to_N(noise_final,ratio)
    signal_bruitee=sum([noise_final,signal])
    return noise_final, signal_bruitee

def noise_gen(path_root, path_babble, l_train_all, l_test_all, SNR, name_snr):
    warnings.filterwarnings("ignore")

    print("Generating data with an SNR of: "+name_snr)
    if not os.path.exists(os.path.join('train',name_snr)):
        os.mkdir(os.path.join('train',name_snr))
    if not os.path.exists(os.path.join('test',name_snr)):
        os.mkdir(os.path.join('test',name_snr))

    train_test = [os.path.join('train','ori'), os.path.join('test','ori')]
    train_test_n = [os.path.join('train',name_snr,'noisy'),os.path.join('test',name_snr,'noisy')]
    if not os.path.exists(train_test_n[0]):
        os.mkdir(train_test_n[0])
    if not os.path.exists(train_test_n[1]):
        os.mkdir(train_test_n[1])

    train_test_no = [os.path.join('train',name_snr,'noise_only'),os.path.join('test',name_snr,'noise_only')]
    if not os.path.exists(train_test_no[0]):
        os.mkdir(train_test_no[0])
    if not os.path.exists(train_test_no[1]):
        os.mkdir(train_test_no[1])

    #Train
    print("Generating train data")
    path_n = os.path.join(path_root,train_test_n[0])
    path_no = os.path.join(path_root,train_test_no[0])
    path_ori = os.path.join(path_root, train_test[0])
    
    for i,names_path in enumerate(tqdm(l_train_all)):
        _, y = read(names_path)
        y = y.astype(np.float32) #Pour l'écriture des wav, scipy accepte les int16,int32 ou float32
        names = os.path.split(names_path)[1]
        noise = random_part_noise(path_babble,len(y))
        noise_final, signal_bruitee = normalisation_and_sum_noise(y,noise,SNR=SNR)

        norm = signal_bruitee.max()-signal_bruitee.min()
        #On normalise tous les sons par rapport au signal bruitee (car snr est sensible aux facteurs de normalisation)
        signal_bruitee_norm = signal_bruitee/norm
        noise_norm = noise_final/norm
    
        #Save noisy sound
        name = names.split('.')
        name_final = name[0]+'_n.'+name[1]
        scipy.io.wavfile.write(os.path.join(path_n,name_final),16000,signal_bruitee_norm)

        #Save noise only
        name = names.split('.')
        name_final = name[0]+'_no.'+name[1]
        scipy.io.wavfile.write(os.path.join(path_no,name_final),16000,noise_norm)
        
    #Test
    path_n = os.path.join(path_root,train_test_n[1])
    path_no = os.path.join(path_root,train_test_no[1])
    path_ori = os.path.join(path_root, train_test[1])

    print("Generating test data")
    for i,names_path in enumerate(tqdm(l_test_all)):
        _, y = scipy.io.wavfile.read(names_path)
        names = os.path.split(names_path)[1]
        noise = random_part_noise(path_babble,len(y))
        noise_final, signal_bruitee = normalisation_and_sum_noise(y,noise,SNR=SNR)

        norm = signal_bruitee.max()-signal_bruitee.min()
        #On normalise tous les sons par rapport au signal bruitee (car snr est sensible aux facteurs de normalisation)
        signal_bruitee_norm = signal_bruitee/norm
        y_norm = y/norm
        noise_norm = noise_final/norm
        
        #Save ori sound
        scipy.io.wavfile.write(os.path.join(path_ori,names),16000,y_norm)
        
        #Save noisy sound
        name = names.split('.')
        name_final = name[0]+'_n.'+name[1]
        scipy.io.wavfile.write(os.path.join(path_n,name_final),16000,signal_bruitee_norm)

        #Save noise only
        name = names.split('.')
        name_final = name[0]+'_no.'+name[1]
        scipy.io.wavfile.write(os.path.join(path_no,name_final),16000,noise_norm)
