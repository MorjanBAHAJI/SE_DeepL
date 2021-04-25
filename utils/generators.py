import keras
import os
import numpy as np
from utils.data_transfo import n_to_any

class DataGenerator(keras.utils.Sequence):
    def __init__(self, l_noisy, batch_size, masktype, mode, output, phase, SNR):
        self.batch_size = batch_size
        if not isinstance(phase, (bool)):
            raise ValueError("phase should be a boolean")
        self.phase=phase
        
        self.l_noisy=l_noisy #liste totale de tous les sons bruités
        self.masktype=masktype
        #Valeurs déterminées manuellement en parcourant tous les spectro en dB (en train et test)
        self.max = 73.94335
        self.min = -49.70791
        if mode!='train' and mode!='test':
            raise ValueError("Please choose valid mode: test or train")
        self.mode=mode
        self.SNR = SNR
        self.output=output
        self.path_no=os.path.join(os.getcwd(),self.mode,self.SNR,'noise_only')
        if not os.path.exists(self.path_no):
            print(f'Path: {self.path_no} \ndo not exists, please check gen_noise.py')
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.l_noisy) / self.batch_size))

    def __getitem__(self, index):
      #index contient la liste des index mais mélangé.
      #Si on récupère une batch_size=5 alors par exemple indexes = [10, 15, 5, 17, 22] 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # On trouve la correspondance des noms 
        l_names_noisy = [] #liste des noms
        #ex: l_names_noisy = ['1004_n.wav', '3_n.wav', ...., '152_n.wav] (de taille batch_size)
        for i in indexes:
            l_names_noisy.append(self.l_noisy[i]) 
        # Generation de batch de taille batch_size
        if self.phase:
            X, Y, P_ori, P_n = self.__gen_data(l_names_noisy)
            return X, Y, P_ori, P_n, l_names_noisy
        else:
            X, Y, _, _ = self.__gen_data(l_names_noisy)
            return X, Y
        
    def on_epoch_end(self):
      #Recupère la liste des indice, et les mélanges pour avoir des batchs différentes d'une epoch à l'autre
        self.indexes = np.arange(len(self.l_noisy))
        np.random.shuffle(self.indexes)

    def __gen_data(self, l_names_noisy): 
        # Generation de batch de taille batch_size
        X=[]
        Y=[]
        P=[]
        #Prend en entrée la liste des son bruité, et donne en sortie les "output" voulu
        params = {'l_names_noisy': l_names_noisy,
        'train_test': self.mode,
        'masktype':self.masktype,
        'output': self.output,
        'path_no': self.path_no
        }
        X,Y,P_ori,P_n = n_to_any(**params)
        return np.array(X), np.array(Y), np.array(P_ori), np.array(P_n)
    
