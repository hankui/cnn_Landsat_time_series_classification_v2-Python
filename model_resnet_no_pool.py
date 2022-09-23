
# import model_resnet_no_pool

# tested on Dec 30, 2021, first batch norm and then relu is better than first relu and then batch norm
# tested on Dec 30, 2021 that batch norm should be applied before softmax on simulated data 

import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, Conv2D, Activation
from tensorflow.keras.layers import Dense, Softmax, Add, MaxPool2D, AveragePooling2D, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Model

neuron_n_4layer  = [64, 128, 256]
neuron_n_8layer  = [64, 128, 256, 256, 512, 512, 1024]
neuron_n_19layer = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 1024,1024]
padding = 'valid'

# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/#:~:text=def%20identity_block(x,relu%27)(x)%0A%20%20%20%20return%20x
def identity_block2d(x, filter0, kernel=(3,3), reg=None):
    # copy tensor to variable called x_skip
    initializer = tf.keras.initializers.HeNormal()
    x_skip = x
    # Layer 1
    x = Conv2D(filter0, kernel, padding='same', kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization(axis=[1,2,3])(x)
    x = Activation('relu')(x)
    
    # Layer 2
    x = Conv2D(filter0, kernel, padding='same', kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization(axis=[1,2,3])(x)
    
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x

def convolutional_block2d(x, filter0, kernel=(3,3), reg=None):
    # copy tensor to variable called x_skip
    initializer = tf.keras.initializers.HeNormal()
    x_skip = x
    # Layer 1
    x = Conv2D(filter0, kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization(axis=[1,2,3])(x)
    x = Activation('relu')(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(filter0, kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg)(x_skip)
    x_skip = BatchNormalization(axis=[1,2,3])(x_skip)
    # Layer 2
    kernel,pool_size,strides = get_kernel_size(x)
    x = Conv2D(filter0, kernel, padding='same', kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False)(x)
    x = BatchNormalization(axis=[1,2,3])(x)
    # Add Residue
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x

def get_kernel_size(x):
    kernel = [3,3]
    pool_size = [2,2]
    strides   = [2,2]
    if x.shape[1]<3:
        kernel[0] = 1
    if x.shape[2]<3:
        kernel[1] = 1
    if x.shape[1]<2:
        pool_size[0] = 1
        strides  [0] = 1
    if x.shape[2]<2:
        pool_size[1] = 1
        strides  [1] = 1  
    return kernel,pool_size,strides

# IMG_HEIGHT=13; IMG_WIDTH=3; IMG_BANDS=1; layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=4; num_classes=2; L2=0; is_batch=True
# layer_n=8; num_classes=2; L2=0; is_batch=True
def ResNet34_no_pool (IMG_WIDTH=3,IMG_HEIGHT=13,IMG_BANDS=1,layer_n=4,num_classes=2,L2=0,is_batch=True):
    ## parameters nitrate
    neuron_n = [64, 128, 256, 512]
     
    block_layers = [1, 1, 1, 1]
    if (layer_n<=4):
        block_layers = [1, 1, 1, 1]
    elif (layer_n<=8):
        block_layers = [2, 2, 2, 2]
    else:
        block_layers = [3, 4, 6, 3]
    
    initializer = tf.keras.initializers.HeNormal()
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    ## build model
    i=0; kernel=[3,3]
    inputs = Input(shape=(IMG_WIDTH,IMG_HEIGHT,IMG_BANDS,))
    x = Conv2D(neuron_n[i], kernel, padding=padding, kernel_initializer=initializer, kernel_regularizer=reg, use_bias=not is_batch)(inputs)
    if (is_batch):
        # x = BatchNormalization()(x)
        x = BatchNormalization(axis=[1,2,3])(x) # fixed on Jan 21 2022
    
    x = Activation('relu')(x)
    for i in range(len(block_layers)):
        print(neuron_n[i])
        kernel,pool_size,strides = get_kernel_size(x)
        # if i==0:
            # For sub-block 1 Residual/Convolutional block not needed
            # for j in range(block_layers[i]):
                # x = identity_block2d(x, neuron_n[i], kernel, reg)
        # else:
        # One Residual/Convolutional Block followed by Identity blocks
        # The filter size will go on increasing by a factor of 2
        # filter_size = filter_size*2
        x = convolutional_block2d(x, neuron_n[i], kernel, reg=reg)
        # x = identity_block2d(x, neuron_n[i], kernel, reg=reg)
        # if i<len(block_layers)-1 or block_layers[i]>1:
            # x = MaxPool2D(pool_size=pool_size,strides=strides,padding='same')(x)
        
        kernel,pool_size,strides = get_kernel_size(x)
        for j in range(block_layers[i] - 1):
            x = identity_block2d(x, neuron_n[i], kernel, reg=reg)
        
        print(x.shape)
    
    # kernel,pool_size,strides = get_kernel_size(x)
    # x = AveragePooling2D(strides, padding = 'same')(x)
    # xf = Flatten()(x)
    xf = GlobalAveragePooling2D()(x)
    
    # fully connected 
    # fln = 512
    # xf = Dense(fln,kernel_initializer=initializer,kernel_regularizer=reg,use_bias=not is_batch)(xf)   
    # if (is_batch):
        # xf = BatchNormalization()(xf)
    # xf = Activation('relu')(xf)
    
    xf = Dense(num_classes,kernel_initializer=initializer, kernel_regularizer=reg)(xf)
    
    # This help regularization 
    # if (is_batch):
        # xf = BatchNormalization()(xf)
    
    output = xf 
    
    # print (model.summary())
    print ("This is a regularized model with L2 = "+str(L2) )
    return Model(inputs, output)

