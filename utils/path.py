import os

def get_n_path(SNR,mode):
    l_noisy = []
    path = os.path.join(os.getcwd(),mode,SNR,'noisy')
    l_total = os.listdir(path)
    [l_noisy.append(os.path.join(os.getcwd(),mode,SNR,'noisy',item)) for item in l_total]     
    return l_noisy


def path_n_to_any(name_n,path_no=None):
    name = os.path.split(name_n)[1].split('_')[0]
    if path_no==None:
        return name+'.wav'
    else:
        name_no = os.path.join(path_no,name+'_no.wav')
        return name+'.wav', name_no
