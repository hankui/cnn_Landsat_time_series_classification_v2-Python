
# import customized_train

import math 
import numpy as np 
import logging

import tensorflow as tf 
import train_test


@tf.function
def loss_class_back(model, x, y, training=True):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_object(y_true=y, y_pred=y_)

# x = input_images_train_norm1[:100,:,:,:IMG_BANDS2]      
# x = input_images_train_norm2[:100,:,:,:IMG_BANDS2]      
# y = y_test[:100]
@tf.function
def loss_class(model, x, y, training=True):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    N_CLASS = y_.shape[1]
    one_hot_y = tf.one_hot(y,N_CLASS)
    y_conv = tf.nn.softmax(y_)   
    # loss1 = -1*tf.math.log(y_conv)*one_hot_y
    # return tf.math.reduce_mean(loss1)
    multi = tf.math.multiply(tf.math.multiply(tf.math.log(y_conv),one_hot_y),-1)
    loss1_sum = tf.math.reduce_sum (multi,axis=1)
    return tf.math.reduce_mean(loss1_sum)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # return loss_object(y_true=y, y_pred=y_)

@tf.function
def loss_reg(model, x, y, training=True):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # y_ = model(x, training=training)
    y_ = tf.reshape(model(x, training=training),y.shape ) ## a bug on Dec 29 2020 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mse = tf.keras.losses.MSE(y_true=y, y_pred=y_)
    # return tf.math.reduce_sum(mse)
    return tf.math.sqrt(tf.math.reduce_mean(mse)) ## changed on Dec 31 2020 & fixed 

# >>> inputs=x
# >>> targets=y
# https://keras.io/guides/writing_a_training_loop_from_scratch/
# https://stackoverflow.com/questions/63959377/keras-regularization-custom-loss
@tf.function
def grad(model, inputs, targets, reg=False, mask_value=math.nan):
    with tf.GradientTape() as tape:
        if reg:
            if math.isnan(mask_value): 
                loss_value = loss_reg(model, inputs, targets, training=True)
                # loss_value = loss_reg(model, inputs, targets, training=None)
            else: 
                # print('test')
                y_ = tf.reshape(model(inputs, training=True),targets.shape )
                mask = tf.cast(targets!=mask_value, y_.dtype)
                targets = tf.cast(targets, y_.dtype)
                abs_vec = tf.multiply(tf.math.square(tf.abs(y_-targets)), mask)
                # abs_vec = tf.multiply(abs_vec
                loss_value = tf.math.sqrt(tf.reduce_sum(abs_vec)/tf.reduce_sum(mask))
                # abs_vec = tf.multiply(tf.abs(y_-targets), mask)
        else:
            loss_value = loss_class(model, inputs, targets, training=True)
            # loss_value += sum(model.losses) ## bug fixed on Jan 20, 2022
            loss_value += sum(model.losses)/2 ## bug fixed on Jan 21, 2022 as use /2 so that the loss is consistent with tf.nn.l2_loss
            # see d:\mycode.RT\mycode.research\cloud_SMART\Pro_test_L2loss.py
            # https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

PLAT_N = 20
PLAT_N = 30
# PLAT_N = 40
VALIDATION_STEP = 50
# @tf.function
def is_validation_flat(validation_accuracies,reg=False):
    n = len(validation_accuracies)
    sub_n = math.ceil(PLAT_N/2)
    last_half = validation_accuracies[(n-sub_n):n]
    penultimate_half = validation_accuracies[(n-2*sub_n):(n-sub_n)]
    if reg:
        return np.array(penultimate_half).mean() < np.array(last_half).mean() # for rmse 
    else:
        return np.array(penultimate_half).mean() > np.array(last_half).mean() # for accuracy 

MIN_FIRST_EPOCH = 25 # This is similar to the one used in R model 
MIN_DELTA_EPOCH = 2
# MIN_FIRST_EPOCH = 15 # This is used for resnet model - fast coverage 
# MIN_DELTA_EPOCH = 2
CHANGE_BASE = 10

# input_images_train_norm2,input_images_test_norm2
## training with validation data 
# @tf.function
# reg=F,mask_value=math.nan
# X_test = input_images_test_norm2
# def trainings_val(model,input_images_train_norm2,input_images_test_norm2,y_train,y_test,LEARNING_RATE,BATCH_SIZE,num_epochs=EPOCH):
def trainings_val(model,X_train,X_test,y_train,y_test,LEARNING_RATE,BATCH_SIZE,num_epochs=100):
    # print ("trainings_val")
    ## training and testing data preparation 
    X_train_sub, y_train_sub, X_validation, y_validation,training_index,validation_index = \
        train_test.random_split_train_validation (X_train=X_train,y_train=y_train,pecentage = 0.04)
    
    train_n = X_train_sub.shape[0]
    train_ds = tf.data.Dataset.from_tensor_slices( (X_train_sub, y_train_sub)).shuffle(train_n+1).batch(BATCH_SIZE)
    test_ds  = tf.data.Dataset.from_tensor_slices( (X_test, y_test)).batch(y_test.shape[0])
    validation_ds  = tf.data.Dataset.from_tensor_slices( (X_validation, y_validation)).batch(y_validation.shape[0])
    
    ## learning rate initialization 
    optimizers = list()
    opt = tf.keras.optimizers.SGD
    max_change_N = 5
    if LEARNING_RATE<0.01:
        max_change_N = 4
    for i in range(max_change_N):
        optimizers.append (opt(learning_rate=LEARNING_RATE/CHANGE_BASE**i, momentum=0.9))
    
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    
    VALIDATION_STEP = math.ceil(len(list(train_ds))/5.1)
    VALIDATION_STEP = math.ceil(len(list(train_ds))/4.1)
    VALIDATION_STEP = math.ceil(len(list(train_ds))/10.1)
    print_str = 'VALIDATION_STEP = ' + str(VALIDATION_STEP)
    logging.info(print_str)
    print(print_str)
    
    # rmse = 0
    totali = 0
    which_rate = 0
    last_epochi = 0
    optimizer = optimizers[which_rate]
    validation_accuracies = []
    validation_epoches = []
    testing_accuracies = []
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    max_change_rate = len(optimizers)
    # rmse_val = 1.0
    val_accuracy = -1
    for epoch in range(num_epochs):                
        # Training loop - using batches
        for x, y in train_ds:
            # break
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_accuracy.update_state(y, model(x, training=True))
            train_loss_results.append(loss_value)
            train_accuracy_results.append(epoch_accuracy.result())
            
            # Track progress
            totali = totali+1
            if (totali%VALIDATION_STEP==0):
                for xv, yv in validation_ds:
                    # break
                    yv = yv.numpy()
                    predicted = model.predict(xv)
                    classesi = np.argmax(predicted,axis=1).astype(np.uint8)
                    val_accuracy = (classesi==yv).sum()/yv.size*100
                
                # if np.isnan(rmse_val) or np.isinf(rmse_val):
                    # return rmse_val,-1,-1
                
                # print_str = '\ttotali batch i ' + str(totali) + " validation accuracy = " + '{:5.3f}'.format(val_accuracy)
                # print(print_str)
                # logging.info(print_str)
                validation_accuracies.append (val_accuracy)
                validation_epoches.append(totali*BATCH_SIZE/X_train_sub.shape[0])
                currenti = len(validation_accuracies)
                current_epoch_i = epoch
                # if which_rate<(max_change_rate-1) and currenti>PLAT_N and is_validation_flat(validation_accuracies):
                if which_rate<(max_change_rate-1) and (current_epoch_i-last_epochi)>MIN_DELTA_EPOCH and epoch>MIN_FIRST_EPOCH and is_validation_flat(validation_accuracies):
                # if which_rate<(max_change_rate-1) and (currenti-lasti)>PLAT_N*2 and is_validation_flat(validation_accuracies):
                    which_rate = which_rate+1
                    last_epochi = epoch
                    # lasti = currenti
                    optimizer = optimizers[which_rate]
                    print_str = "Note that the learning rate is reduced to " + str(LEARNING_RATE/CHANGE_BASE**(which_rate)) + 'at validation step ' + str(currenti)
                    print(print_str)
                    logging.info(print_str)
        
        if test_ds!=0 and epoch%5==0:
            for xs, ys in test_ds:
                # break
                ## the following three variables are all different 
                ## model(x) != model(x, training=False)
                ## model(x) != model(x, training=True)
                ys = ys.numpy()
                predicted = model.predict(xs)
                classesi = np.argmax(predicted,axis=1).astype(np.uint8)
                test_accuracy = (classesi==ys).sum()/ys.size*100
            testing_accuracies.append(test_accuracy)
        
        # before end epoch
        if epoch>=5 and val_accuracy<10: ## something this wrong as this should be greater than 10% at this stage 
            return val_accuracy,validation_accuracies,validation_epoches,train_loss_results,train_accuracy_results
            
        if epoch%5==0:
            print_str = "Epoch {:03d}: switch rate {:3d}, Train: {:.3f}, Validation: {:.3f}, Test: {:.3f}".format(epoch, which_rate, epoch_accuracy.result(),val_accuracy,test_accuracy)
            logging.info(print_str)
            print(print_str)
            # print("Inter = {:03d} Intra = {:03d} ", tf.config.threading.get_inter_op_parallelism_threads(), tf.config.threading.get_intra_op_parallelism_threads() )
        
    return val_accuracy,validation_accuracies,validation_epoches,train_loss_results,train_accuracy_results



# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
# y_pred = model.predict(input_images_train_norm1[:100,:,:,:IMG_BANDS2])
# y_true = y_test[:100]
@tf.function
def my_loss(y_pred,y_true):
    one_hot_y = tf.one_hot(y_true,N_CLASS)
    y_conv = tf.nn.softmax(y_pred)   
    # loss1 = -1*tf.math.log(y_conv)*one_hot_y
    # return tf.math.reduce_mean(loss1)
    multi = tf.math.multiply(tf.math.multiply(tf.math.log(y_conv),one_hot_y),-1)
    loss1_sum = tf.math.reduce_sum (multi,axis=1)
    return tf.math.reduce_mean(loss1_sum)
 
def test_accuacy(model,input_images,y_test):
    logits = model.predict(input_images)
    classesi = np.argmax(logits,axis=1).astype(np.uint8).reshape(y_test.shape)
    accuracy = (y_test==classesi).sum()/classesi.size
    return accuracy,classesi
