
# import model

# tested on Dec 30, 2021, first batch norm and then relu is better than first relu and then batch norm
# tested on Dec 30, 2021 that batch norm should be applied before softmax on simulated data 

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, Conv2D, Activation
from tensorflow.keras.layers import Dense, Softmax, Flatten

from tensorflow.keras.models import Model

neuron_n_4layer  = [64, 128, 256]
neuron_n_5layer  = [64, 64, 64, 256]
neuron_n_8layer  = [64, 128, 256, 256, 512, 512, 1024]
neuron_n_19layer = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 1024,1024]
# neuron_n_4layer  = [128, 256, 512]
# neuron_n_4layer  = [32, 64, 128]
# neuron_n_4layer  = [4, 4, 4]
# neuron_n_8layer  = [64, 64, 64, 64, 64, 64, 64]
# neuron_n_8layer  = [64, 64, 64, 64, 64, 256, 256]
# neuron_n_4layer  = [32, 32, 32] 
# neuron_n_8layer  = [32, 32, 32, 32, 32, 32, 32]
first_kernel = [3,9]
first_kernel = [3,7]
# first_kernel  = [3,5] ## better 
first_kernel = [5,3]
second_kernel = [3,3]
# second_kernel = [3,5]
padding = 'same'
padding = 'valid'

# IMG_HEIGHT=13; IMG_WIDTH=3; IMG_BANDS=1; layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=8; num_classes=2; L2=0; is_batch=True
def get_model_cnn_2d (IMG_WIDTH=3,IMG_HEIGHT=13,IMG_BANDS=1,layer_n=4,num_classes=2,L2=0,is_batch=True):
    ## parameters nitrate
    neuron_n = neuron_n_4layer
    if (layer_n<=4):
        neuron_n = neuron_n_4layer
    elif (layer_n<=5):
        neuron_n = neuron_n_5layer
    elif (layer_n<=8):
        neuron_n = neuron_n_8layer
    else:
        neuron_n = neuron_n_19layer
    
    ## build model
    inputs = Input(shape=(IMG_WIDTH,IMG_HEIGHT,IMG_BANDS,))
    
    reg = None
    # padding = 'same'
    # padding = 'valid'
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    for i in range(layer_n-2):
        print(neuron_n[i])
        # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=math.sqrt(2/neuron_n[i]) )
        initializer = tf.keras.initializers.HeNormal()
        kernel = [3,3]
        if i>0: 
            if x.shape[1]<3:
                kernel[0] = 1
            if x.shape[2]<3:
                kernel[1] = 1
        if layer_n==19:
            if i==0:
                x = Conv2D(neuron_n[i], kernel, padding='same', kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(inputs)
            elif i==2 or i==4 or i==5 or i==6 or i==8 or i==9 or i==11 or i==12 or i==15:
                x = Conv2D(neuron_n[i], kernel, padding='same', kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
            else:
                x = Conv2D(neuron_n[i], kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
        else:   
            if i==0:
                x = Conv2D(neuron_n[i], kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(inputs)
            else:
                x = Conv2D(neuron_n[i], kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
        
        if (is_batch):
            # x = BatchNormalization()(x)
            x = BatchNormalization(axis=[1,2,3])(x) # fixed on Jan 21 2022
        
        x = Activation('relu')(x)
        print(x.shape)
    
    # not tested yet
    # xf = tf.reshape(x,(-1,x.shape[1]*x.shape[2]*x.shape[3]))
    xf = Flatten()(x)
    xf = Dense(neuron_n[i+1],kernel_initializer=initializer,kernel_regularizer=reg,use_bias=not is_batch)(xf)   
    if (is_batch):
        xf = BatchNormalization()(xf)
    
    xf = Activation('relu')(xf)
    xf = Dense(num_classes,kernel_initializer=initializer, kernel_regularizer=reg)(xf)
    
    # if (is_batch):
        # xf = BatchNormalization()(xf)
    
    # output = tf.nn.softmax(xf)
    # output = Softmax()(xf)
    output = xf 
    
    # print (model.summary())
    print ("This is a regularized model with L2 = "+str(L2) )
    return Model(inputs, output)

# IMG_HEIGHT=13; IMG_WIDTH=3; layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=8; num_classes=2; L2=0; is_batch=True
def get_model_cnn_1d (IMG_WIDTH=3,IMG_HEIGHT=13,layer_n=4,num_classes=2,L2=0,is_batch=True):
    ## parameters 
    neuron_n = neuron_n_4layer
    if (layer_n<=4):
        neuron_n = neuron_n_4layer
    elif (layer_n<=5):
        neuron_n = neuron_n_5layer
    elif (layer_n<=8):
        neuron_n = neuron_n_8layer
    else:
        neuron_n = neuron_n_19layer
    
    ## build model
    # IMG_BANDS
    inputs = Input(shape=(IMG_WIDTH,IMG_HEIGHT,))
    
    reg = None
    # padding = 'same'
    # padding = 'valid'
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    for i in range(layer_n-2):
        print(neuron_n[i])
        # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=math.sqrt(2/neuron_n[i]) )
        initializer = tf.keras.initializers.HeNormal()
        kernel = 3
        if i>0: 
            if x.shape[1]<3:
                kernel = 1
        
        if layer_n==19:
            if i==0:
                x = Conv1D(neuron_n[i], kernel, activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(inputs)
            elif i==2 or i==4 or i==5 or i==6 or i==8 or i==9 or i==11 or i==12 or i==15:
                x = Conv1D(neuron_n[i], kernel, activation='relu', padding='same', kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
            else:
                x = Conv1D(neuron_n[i], kernel, activation='relu', padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
        else:   
            if i==0:
                x = Conv1D(neuron_n[i], kernel, activation='relu', padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(inputs)
            else:
                x = Conv1D(neuron_n[i], kernel, activation='relu', padding=padding, kernel_initializer=initializer, kernel_regularizer=reg,use_bias=not is_batch)(x)
        
        if (is_batch):
            # x = BatchNormalization()(x)
            x = BatchNormalization(axis=[1,2])(x) # fixed on Jan 21 2022
        
        # x = Activation('relu')(x)
        print(x.shape)
    
    # not tested yet
    xf = tf.reshape(x,(-1,x.shape[1]*x.shape[2]))
    xf = Dense(neuron_n[i+1], activation='relu',kernel_initializer=initializer,kernel_regularizer=reg,use_bias=not is_batch)(xf)
    
    if (is_batch):
        xf = BatchNormalization()(xf)
    
    # xf = Activation('relu')(xf)
    xf = Dense(num_classes,kernel_initializer=initializer, kernel_regularizer=reg)(xf)
    
    # if (is_batch):
        # xf = BatchNormalization()(xf)
    
    # output = tf.nn.softmax(xf)
    # output = Softmax()(xf)
    output = xf 
    
    # print (model.summary())
    print ("This is a 1D regularized model with L2 = "+str(L2) )
    return Model(inputs, output)
