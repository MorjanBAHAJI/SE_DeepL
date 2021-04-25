import sys
import os
from time import sleep
from tqdm import tqdm
from utils.noise import noise_gen

if __name__ == '__main__':
    l = sys.argv[1:]
    for num in l:
        try:
            num = int(num)
        except ValueError:
            print("Oops! "+num+" is not a valid SNR" )
            exit(1)
            
        path_train = os.path.join('train','ori')
        path_test = os.path.join('test','ori')
        files_train = os.listdir(path_train)
        files_test = os.listdir(path_test)
        l_train = [os.path.join(os.getcwd(),path_train,name) for name in files_train]
        l_test =[os.path.join(os.getcwd(),path_test,name) for name in files_test]
        
        dict_noise = {'path_root' : os.getcwd(),
                      'path_babble':'babble_sub.wav',
                      'l_train_all': l_train,
                      'l_test_all': l_test,
                      'SNR': num,
                      'name_snr': str(num)+'dB'
                      }

        noise_gen(**dict_noise)
