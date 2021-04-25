from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from utils.generators import DataGenerator
from utils.path import *
from utils.models import model1, model2, unet
from utils.norm import unorm_to_audio
from utils.data_transfo import get_ori, get_spectro
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import IPython
import re

##DISPLAY FUNCTIONS
####################################################################################################
def display_results(model,name_ori,output,SNR, X_in, Y_out,masktype):
    if output=='mask':
        output = masktype+' '+output
        
    for i,sound in enumerate(Y_out):
        print(name_ori[i]+' noised with an input SNR of: ', SNR,'...')
        add_display(X_in[i])
        print(name_ori[i]+' denoised with '+ model+' and '+output)
        add_display(sound)

        
def plot_waveform(l_noisy,masktype,SNR_, SNR_load,output,model,mode):
    names, X_in,X_ori, Y_out = test(l_noisy,masktype,SNR_, SNR_load,output,model,mode)
    for i in range(len(l_noisy)):
        plt.plot(X_in[i],label='Noised sound')
        plt.plot(Y_out[i],label='Denoised sound')
        plt.title("Sound: "+ path_n_to_any(names[i])+ " with "+model+" "+output+" input SNR of: "+SNR_)
        plt.ylabel('Normalized amplitude')
        plt.legend()
        plt.xlabel('Ech')
        plt.show()
            
def add_display(data):
    IPython.display.display(IPython.display.Audio(data,rate=16000))
    
    
    
def print_res_metric(SNR_final,MSE_final,MAE_final,model, masktype,output,SNR_):
    SNR_final_Gain = SNR_final - float(re.findall(r'-?\d+', SNR_)[0])
    if output=='mask':
        output = masktype+' '+output
    a = f'For {model} and an output {output}'
    b =  f' with an input SNR of {SNR_},\n'
    c = f'--->The output mean Gain in SNR (after denoising) is : {round(SNR_final_Gain,1)}\n'
    d = f'--->The output mean MAE (after denoising) is : {MAE_final}\n'
    e = f'--->The output mean MSE (after denoising) is : {MSE_final}\n'
    print(a+b+c+d+e)
    
    
def plot_spectro(output,l_name,masktype,SNR_,SNR_load,model,mode):
    names, X_in,X_ori, Y_out = test(l_name,masktype,SNR_, SNR_load,output,model,mode,save='none')
    name_ori = [path_n_to_any(name) for name in names]
    if output=='mask':
        output = masktype+' '+output
    
    print('Spectrograms: '+model+' with '+ output+ ' and an input SNR = '+SNR_)
    for i,sound in enumerate(X_in):
        mag_n,_=get_spectro(sound, dB=True)
        mag_final,_=get_spectro(Y_out[i], dB=True)
        mag_ori,_=get_spectro(X_ori[i], dB=True)
        
        fig = plt.figure(figsize=(20,3))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(mag_n)
        ax.set_title("Noised sound: "+name_ori[i])
        ax = fig.add_subplot(1,3,2)
        ax.imshow(mag_final)
        ax.set_title("Denoised sound ")
        ax = fig.add_subplot(1,3,3)
        ax.imshow(mag_final)
        ax.set_title("Original sound: ")
    
####################################################################################################


###METRICS FUNCTIONS
####################################################################################################

def snr(x_ori, sound):
    P_ori = sum(x_ori**2)
    P_sub = sum((sound - x_ori)**2)
    return 10*np.log10(P_ori/P_sub)

def mean_metric_total(l_noisy,masktype,SNR_,SNR_load,output,model,mode,save='none',get_perf=False):
    print('For all',len(l_noisy), 'batches...')
    snr_tab = []
    mae_tab = []
    mse_tab = []
    
    for l_name in tqdm(l_noisy):
        _,X_in,X_ori, Y_out = test(l_name,masktype,SNR_,SNR_load,output,model,mode,get_perf=get_perf)
        snr, mse, mae = metric(Y_out, X_in, X_ori, mean=True) # Mean over the current batch
        X_in.clear()
        X_ori.clear()
        Y_out.clear()
        snr_tab.append(snr)    
        mae_tab.append(mae)    
        mse_tab.append(mse)    
    return round(sum(snr_tab)/len(snr_tab),1),round(sum(mse_tab)/len(mse_tab),5) ,round(sum(mae_tab)/len(mae_tab),5) #Mean over all batches

def metric(Y_out, X_in, X_ori, mean=True):
    snr_tab = []
    mae_tab = []
    mse_tab = []
    for i,sound in enumerate(Y_out):
        snr_tab.append(snr(X_ori[i], sound))
        mae_tab.append(tf.keras.losses.MAE(X_ori[i],sound).numpy())
        mse_tab.append(tf.keras.losses.MSE(X_ori[i],sound).numpy())
    if mean:
        return sum(snr_tab)/len(snr_tab),sum(mse_tab)/len(mse_tab) ,sum(mae_tab)/len(mae_tab) 
    else:
        return snr_tab, mse_tab, mae_tab
    
    
def get_batch(lim,batch_,SNR_, mode):
    l_noisy = get_n_path(SNR_,mode)
    if lim==-1:
    	lim=len(l_noisy)
    np.random.shuffle(l_noisy)
    batch = []
    div = lim//batch_ 
    reste = lim%batch_
    
    if batch_==1:
        return l_noisy[0:lim]
    
    if lim >len(l_noisy):
        raise ValueError("Please choose a lim value smaller than: ",len(l_noisy))
    else:
        for i in range(0,div):
            batch.append(l_noisy[i*batch_:(i+1)*batch_])
        if reste!=0:
            batch.append(l_noisy[div*batch_:(div*batch_)+ reste])
    return batch
####################################################################################################

###TRAIN AND TEST FUNCTIONS
####################################################################################################

def define_and_load(m_name,masktype,output,SNR,SNR_load):
    models_names = ['model1','model2', 'unet']
    if m_name=='unet':
        name = models_names[2]
        model = unet(output=output)
    elif m_name=='model2':
        name = models_names[1] 
        model = model2(output=output)
    elif m_name=='model1':
        name = models_names[0] 
        model = model1(output=output)
    else:
        raise ValueError("Please choose 'unet', 'model1' or 'model2'")

    if masktype:
            model_name = 'models/'+name+'_n_to_'+output+'_'+masktype+'_'+SNR
    else:
        model_name = 'models/'+name+'_n_to_'+output+'_'+SNR        
    
    loss = 'mean_squared_error'
    if output=='mask' and masktype=='Binary':
        metrics = tf.keras.metrics.BinaryAccuracy()
    else:
        metrics = tf.keras.metrics.MeanAbsoluteError()  

    model.compile(optimizer=Adam(lr=1e-3), 
                loss=loss, 
                metrics=[metrics])
    if SNR_load:
        if masktype:
            model.load_weights('models/'+name+'_n_to_'+output+'_'+masktype+'_'+SNR_load+'.h5')
        else:
            model.load_weights('models/'+name+'_n_to_'+output+'_'+SNR_load+'.h5')

    return model, model_name
    


def init_model_and_train(m_name,output,SNR,epoch,l_dict,masktype=None,SNR_load=None):
    dict_train = l_dict[0]
    dict_valid = l_dict[1]

    model, model_name = define_and_load(m_name,masktype,output,SNR,SNR_load)
    callbacks = [ModelCheckpoint(model_name+'.h5' , verbose=1, save_best_only=True)]

    train_generator = DataGenerator(**dict_train)
    valid_generator = DataGenerator(**dict_valid)

    history = model.fit(train_generator, validation_data=valid_generator, epochs=epoch,callbacks=callbacks)
    dict_res = history.history
    df = pd.DataFrame.from_dict(dict_res)
    df.to_csv(model_name+'.csv', index=False)
    df.head()



def train(output,SNR,epoch,net,lim=None,SNR_load=None, masktype=None, phase=False):
    mode = 'train'
    output = output # What we want to predict: 'ori' (original), 'mask' (mask)
    if masktype:
        masktype = masktype # Binary or Soft
    phase = phase #Put on True to get the Phase at the output, False otherwise
    SNR =SNR

    l_total = get_n_path(SNR,mode)
    l_noisy_train=l_total[:round(0.75*len(l_total))]
    l_noisy_valid=l_total[round(0.25*len(l_total)):]
    if lim:
        l_noisy_train=l_total[0:round(lim)]
        l_noisy_valid=l_total[0:round(0.25*lim)]

    dict_train = {'l_noisy': l_noisy_train,
            'batch_size': 5,
            'masktype':masktype,
            'mode': mode,
            'output': output,
            'phase': phase,
            'SNR': SNR
            }

    dict_valid = {'l_noisy': l_noisy_valid,
            'batch_size': 5,
            'masktype':masktype,
            'mode': mode,
            'output': output,
            'phase': phase,
            'SNR': SNR
            }
    l_dict = [dict_train, dict_valid]
    if net=='all':
        print("Training model2....")
        init_model_and_train('model2',output,SNR,epoch,l_dict,masktype=masktype,SNR_load=SNR_load)
        print("Training unet....")
        init_model_and_train('unet',output,SNR,epoch,l_dict,masktype=masktype,SNR_load=SNR_load)
    elif net=='unet':
        print("Training unet....")
        init_model_and_train('unet',output,SNR,epoch,l_dict,masktype=masktype,SNR_load=SNR_load)
    elif net=='model2':
        print("Training model2....")
        init_model_and_train('model2',output,SNR,epoch,l_dict,masktype=masktype,SNR_load=SNR_load)
    elif net=='model1':
        print("Training model1....")
        init_model_and_train('model1',output,SNR,epoch,l_dict,masktype=masktype,SNR_load=SNR_load)
    else:
        raise ValueError("Please choose valid net: 'all', 'unet', 'model1' or 'model2'")

        
def test(l_noisy,masktype,SNR,SNR_load,output,model,mode,save='none',get_perf=False):
    X_in = []
    Y_out = []
    X_ori = []
    
    dict_test = {'l_noisy': l_noisy,
            'batch_size': len(l_noisy),
            'masktype':masktype,
            'mode': mode,
            'output': output,
            'phase': True,
            'SNR': SNR
            }
    X,Y,P_ori,P_n,l_names_noisy= DataGenerator(**dict_test).__getitem__(0)
    
    for name_n in l_names_noisy: 
        __, name_no= path_n_to_any(name_n,path_no=os.path.join(os.getcwd(),mode,SNR,'noise_only'))
        X_ori.append(get_ori([name_n, name_no]))
        
    params = {'X': X,
    'Y': '',
    'P_ori':P_ori,
    'P_n': P_n,
    'output': output,
    'masktype': masktype,
    'l_noisy': l_names_noisy,
    'net': '',
    'SNR': SNR,
    'save': save
    }
    
    
    if model not in ['all', 'unet', 'model1', 'model2']:
        raise ValueError("Please choose valid model: 'all', 'unet', 'model1' or 'model2'")
    
    if not get_perf:
        model, _ = define_and_load(model,masktype,output,SNR,SNR_load)
        Y_predict = model.predict(X)
        params["Y"] = Y_predict
    else:
        params["Y"] = Y
    params["net"] = model
    x_in, y_out = unorm_to_audio(**params)    
    return l_names_noisy, x_in, X_ori, y_out

####################################################################################################
    
