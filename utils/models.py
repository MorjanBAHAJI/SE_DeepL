from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.regularizers import l2


def bloc_conv(nbconv, x, n_filters, kernel_size = 3, batchnorm = False, activation='relu',padding = 'same'):
    for i in range(nbconv):
        if batchnorm:
            x = Conv2D(filters = n_filters, kernel_size = (kernel_size,kernel_size),
                       activation=activation, kernel_initializer = 'he_normal', 
                       padding = padding)(BatchNormalization()(x))
        else:
            x = Conv2D(filters = n_filters, kernel_size = (kernel_size,kernel_size),
                       activation=activation, kernel_initializer = 'he_normal', 
                       padding = padding)(x)
    return x

def model1(output='mask'):
    inputs = Input((129, 385,1))
    #Premier bloc
    x1 = bloc_conv(2, inputs, 64, kernel_size = 3) 
    x = bloc_conv(2, inputs, 64, kernel_size = 3)
    x = MaxPooling2D(pool_size=(2, 2),padding="same")(x)
    
    #Second bloc
    x = bloc_conv(3, x, 128, kernel_size = 3, batchnorm=True)
    x = bloc_conv(3, x, 64, kernel_size = 3, batchnorm=True)
    x = UpSampling2D(size=(2, 2))(x)
    
    #Troisème bloc
    x = Cropping2D(cropping=((1, 0),(1, 0)))(x)
    merge1 = concatenate([x, x1], axis=3)
    x = bloc_conv(2, merge1, 32, kernel_size = 3, batchnorm=True)
    
    if output=='mask':
        activation = 'sigmoid'
    elif output=='ori':    
        activation = 'tanh'
    else:
        raise ValueError("None valid output")
    
    xo = bloc_conv(1, x, 1, kernel_size = 3, activation=activation)

    model = Model(inputs, xo)
    return model

def model2(output='mask'):
    skip = []
    inputs = Input((129, 385,1))
    x = bloc_conv(2, inputs, 64, kernel_size = 3) 
    skip.append(x)
    
    x = MaxPooling2D(pool_size=(2, 2),padding="same")(x) #DIV 2
    x = bloc_conv(2, x, 128, kernel_size = 3,batchnorm = True) # skiped
    skip.append(x)
    
    x = MaxPooling2D(pool_size=(2, 2),padding="same")(x) #DIV 4
    x = bloc_conv(2, x, 256, kernel_size = 3,batchnorm = True) 
    skip.append(x)
    
    x = MaxPooling2D(pool_size=(2, 2),padding="same")(x) #DIV 8
    x = bloc_conv(2, x, 512, kernel_size = 3,batchnorm = True) 
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Cropping2D(cropping=((1, 0),(1, 0)))(x)    
    merge = concatenate([x, skip[2]], axis=3)
    x = bloc_conv(2, x, 256, kernel_size = 3,batchnorm = True) 
    x = Dropout(0.33)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Cropping2D(cropping=((1, 0),(1, 0)))(x)
    merge = concatenate([x, skip[1]], axis=3)
    x = bloc_conv(2, x, 128, kernel_size = 3,batchnorm = True) 

    x = UpSampling2D(size=(2, 2))(x)
    x = Cropping2D(cropping=((1, 0),(1, 0)))(x)
    merge = concatenate([x, skip[0]], axis=3)
    x = bloc_conv(2, x, 64, kernel_size = 3,batchnorm = True) 
    x = bloc_conv(1, x, 32, kernel_size = 3,batchnorm = True)
    
    if output=='mask':
        activation = 'sigmoid'
    elif output=='ori':    
        activation = 'tanh'
    else:
        raise ValueError("None valid output")
    
    xo = bloc_conv(1, x, 1, kernel_size = 3, activation=activation)
    model = Model(inputs, xo)
    return model



def unet(output='mask'):
    inputs = Input((129,385,1))
    
    #Premier bloc
    x1 = bloc_conv(nbconv=2, x = inputs, n_filters=64, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')
    m_pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(BatchNormalization()(x1))
    
    #Second bloc
    x2 = bloc_conv(nbconv=2, x = m_pool1, n_filters=128, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')
    m_pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(BatchNormalization()(x2))
    
    
    #Troisième bloc
    x3 = bloc_conv(nbconv=2, x = m_pool2, n_filters=256, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')    
    m_pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(BatchNormalization()(x3))
    
    #Quatrième bloc    
    x4 = bloc_conv(nbconv=2, x = m_pool3, n_filters=512, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')   
    drop4 = Dropout(0.33)(BatchNormalization()(x4))
    m_pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(BatchNormalization()(drop4))
    
    #Cinquième bloc
    x5 = bloc_conv(nbconv=2, x = m_pool4, n_filters=1024, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')   
    drop5 = Dropout(0.33)(BatchNormalization()(x5))

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')( #Upsamp and conv 2x2
        UpSampling2D(size=(2, 2))(BatchNormalization()(drop5)))
    up6 = Cropping2D(cropping=((1, 0),(1, 0)))(up6)
    
    #Sixième bloc
    concat6 = concatenate([drop4, up6], axis=3)
    x6 = bloc_conv(nbconv=2, x = concat6, n_filters=512, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')   
    
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(BatchNormalization()(x6)))
    up7 = Cropping2D(cropping=((1, 0),(1, 0)))(up7)
    
    #Septième bloc
    concat7 = concatenate([x3, up7], axis=3)
    x7 = bloc_conv(nbconv=2, x = concat7, n_filters=256, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')   
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(BatchNormalization()(x7)))
    up8 = Cropping2D(cropping=((1, 0),(1, 0)))(up8)
    
    #Huitième bloc
    concat8 = concatenate([x2, up8], axis=3)    
    x8 = bloc_conv(nbconv=2, x = concat8, n_filters=128, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same') 

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(BatchNormalization()(x8)))
    up9 = Cropping2D(cropping=((1, 0),(1, 0)))(up9)

    #Neuvième bloc
    concat9 = concatenate([x1, up9], axis=3)
    x9 = bloc_conv(nbconv=2, x = concat9, n_filters=64, kernel_size = 3, 
                      batchnorm = True, activation='relu',padding = 'same')   
    
    x9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BatchNormalization()(x9))
    
    if output=='mask':
        activation = 'sigmoid'
    elif output=='ori':    
        activation = 'tanh'
    else:
        raise ValueError("None valid output")

    
    x9 = Conv2D(1, 1, activation=activation)(BatchNormalization()(x9))
    
    model = Model(inputs, x9)

    return model