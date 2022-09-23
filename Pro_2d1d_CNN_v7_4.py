# v7.4 on Sep 22, 2022 to use 5 metrics & 5 observation threshold & Hank is going to push to github

# date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# python Pro_learn_partial_CNN_v1_5.py
# date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
# date
# echo "$time_diff hours used";

# import datetime
# start_time = datetime.datetime.now()
# import Pro_2d1d_CNN_v6_0
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)

# import datetime
# start_time = datetime.datetime.now()
# import importlib
# importlib.reload(Pro_2d1d_CNN_v2_3)
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)

# module load cuda
# module load cudnn
# module load python/3.7
# module load rasterio
# module load libtiff 
# module load libgeotiff
import os
import logging
import socket
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

import model
# import model_partial
import plot_time_series
from plot_time_series import band_fields

import train_test 
import customized_train

import importlib
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print(socket.gethostname())
base_name = "this_log_"+socket.gethostname()

#*****************************************************************************************************************
## load data
csv_file2 = '/gpfs/scratch/hankui.zhang/workspace/LCMAP/dump.metric.2022.03.04.all.step40.csv'
# new_file = csv_file2.replace('dump','sub.dump7')
new_file = csv_file2.replace('dump','sub.dump5')
class_field = 'nlcd'
n_field2 = 'n_valid'
if not os.path.exists(new_file):
    data_per_all = pd.read_csv(csv_file2)
    yclasses = data_per_all[class_field]
    # valid_index = np.logical_and(yclasses != 12, data_per_all[n_field2]>11)
    valid_index = np.logical_and(yclasses != 12, data_per_all[n_field2]>=5)
    data_per_all[valid_index].to_csv(new_file) 

data_per = pd.read_csv(new_file)
yclasses = data_per[class_field]

# ids = data_per[id_field]
# valid_index = np.logical_and(yclasses != 12, data_per[n_field2]>11)
valid_index = yclasses != 12
# data_per[class_field][data_per[class_field]==12] = 100

#*****************************************************************************************************************
## split training & testing data 
import train_test 
importlib.reload(train_test)
orders = train_test.random_split(data_per.shape[0],split_n=10)
index_train = np.logical_and(orders==0,valid_index)
index_test =  np.logical_and(orders!=0,valid_index)

unique_yclass = np.unique(yclasses)
print (np.unique(yclasses[index_train]))
print (np.unique(yclasses[index_test ]))

def mapping_classes():
    unique_yclass = np.unique(yclasses[valid_index])
    for i in range(unique_yclass.size):
        data_per[class_field][data_per[class_field]==unique_yclass[i]] = i

mapping_classes()
# data_all[class_field][data_all[class_field]==7] = 6
# data_per[class_field][data_per[class_field]==7] = 6
# unique_id = np.unique(ids)
unique_yclass = np.unique(yclasses)
print (np.unique(yclasses[index_train]))
print (np.unique(yclasses[index_test ]))
N_CLASS = np.unique(yclasses[index_train]).size

#*****************************************************************************************************************
## construct training and testing data
## plain CNN model
IMG_HEIGHT1 = 3   ; IMG_WIDTH1=13; IMG_BANDS1=1
IMG_HEIGHT2 = 5   ; IMG_WIDTH2=13; IMG_BANDS2=1
# IMG_HEIGHT2 = 7   ; IMG_WIDTH2=13; IMG_BANDS2=1
IMG_HEIGHT2 = 9   ; IMG_WIDTH2=13; IMG_BANDS2=1
BATCH_SIZE = 128*2; 
LEARNING_RATE = 0.01; layer_n = 4; EPOCH = 5; ITERS = 1; L2 = 1e-3; METHOD=0; PERCENT = 0.1; GPUi = 0

MODEL_DIR = "./model/"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

import sys
if __name__ == "__main__":
    LEARNING_RATE = 0.01
    EPOCH = 200 
    EPOCH = 5
    METHOD = 1 # Adam 
    METHOD = 0 # Hank
    ITERS = 1 
    PERCENT = 0.1
    GPUi = 0
    print ("sys.argv n: " + str(len(sys.argv)))
    LEARNING_RATE = float(sys.argv[1])
    EPOCH         = int(sys.argv[2] )
    METHOD        = int(sys.argv[3] )
    # ITERS         = int(sys.argv[4] )
    L2            = float(sys.argv[4])
    if len(sys.argv)>5:
        layer_n       = int(sys.argv[5])

    if len(sys.argv)>6:
        PERCENT       = float(sys.argv[6])
    
    if len(sys.argv)>7:
        GPUi       = int(sys.argv[7])
    
    if len(sys.argv)>8:
        IMG_HEIGHT2  = int(sys.argv[8])
    
    #*****************************************************************************************************************
    ## set GPU
    if '__file__' in globals():
        base_name = os.path.basename(__file__)+socket.gethostname()
        print(os.path.basename(__file__))
    
    base_name = base_name+'.perc'+str(PERCENT)+'.layer'+str(layer_n)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.LEARNING_RATE'+str(LEARNING_RATE)+'.EPOCH'+str(EPOCH)+'.L2'+str(L2)
    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[GPUi], 'GPU')  
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)    
    
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    # exit()
    #*****************************************************************************************************************
    ## get train and testing data
    importlib.reload(train_test)
    y_train,y_test,\
        input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,\
        input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3, dat_out, mean_train2, std_train2 = \
        train_test.get_training_test7(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = PERCENT)
    
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    
    #*****************************************************************************************************************
    # save mean and std 
    mean_name=MODEL_DIR+base_name+'.mean.csv'
    mean = mean_train2.copy()
    std  = std_train2.copy()
    # !!!!!!! transpose & reshape are different 
    arr = np.concatenate((mean.reshape(1,mean.shape[0]*mean.shape[1]), std.reshape(1,mean.shape[0]*mean.shape[1]) )).transpose()
    header = 'mean,std'
    np.savetxt(mean_name, arr, fmt="%s", header=header, delimiter=",")
    
    #*****************************************************************************************************************
    ## plain CNN
    print ("\n\n#plain CNN *****************************************************************************************************************\n\n\n")
    accuracylist1 = list(); accuracylist2 = list(); accuracylist3 = list()
    testi = 0
    
    #*****************************************************************************************************************
    ## model 3 1d 13 * 5
    print_str = "\n 3: 1d cnn model 13*5 *****************************************************************************************************************"
    print (print_str); logging.info (print_str)
    importlib.reload(model)
    model_cnn = model.get_model_cnn_1d (IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,layer_n=layer_n,num_classes=N_CLASS, L2=L2,is_batch=True) 
    print (model_cnn.summary())
    print (LEARNING_RATE)
    if METHOD==1:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model_cnn.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model_cnn.fit(input_images_train_norm3, y_train, validation_split=0.04, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=2)
    else: 
        importlib.reload(customized_train)
        val_accuracy = -1; times = 0
        while val_accuracy<10 and times <5:
            model_cnn = model.get_model_cnn_1d (IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,layer_n=layer_n,num_classes=N_CLASS, L2=L2,is_batch=True) 
            times+=1
            val_accuracy,validation_accuracies,validation_epoches,train_loss_results,train_accuracy_results  \
                = customized_train.trainings_val(model_cnn,input_images_train_norm3,input_images_test_norm3,y_train,y_test,LEARNING_RATE=LEARNING_RATE,BATCH_SIZE=BATCH_SIZE,num_epochs=EPOCH)
        
        print (validation_accuracies)
        print (validation_epoches)
    
    accuracy,classesi = customized_train.test_accuacy(model_cnn,input_images_test_norm3,y_test)
    print ("cnn" + str(testi) + '  {:0.4f}'.format(accuracy) )
    print ("cnn" + str(testi) + '  {:4.2f}'.format(accuracy*100) )
    accuracylist3.append (accuracy)
    dat_out['predicted_cnn3'] = classesi
    model_name = MODEL_DIR+base_name+'.1d.model_cnn.h5'
    model_cnn.save(model_name) 
    
    #*****************************************************************************************************************
    ## model 4 13 * 5
    # print_str = "\n 4: 2d cnn resnet model 13*5 *****************************************************************************************************************"
    # print (print_str); logging.info (print_str)
    # import model_resnet_no_pool
    # importlib.reload(model_resnet_no_pool)
    # model_cnn = model_resnet_no_pool.ResNet34_no_pool (IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2,layer_n=layer_n,num_classes=N_CLASS, L2=L2,is_batch=True) 
    # print (model_cnn.summary())
    # print (LEARNING_RATE)
    # if METHOD==1:
        # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        # model_cnn.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # model_cnn.fit(input_images_train_norm2, y_train, validation_split=0.04, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=2)
    # else: 
        # importlib.reload(customized_train)
        # customized_train.MIN_FIRST_EPOCH = 20
        # val_accuracy = -1; times = 0
        # while val_accuracy<10 and times <5:
            # model_cnn = model_resnet_no_pool.ResNet34_no_pool (IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2,layer_n=layer_n,num_classes=N_CLASS, L2=L2,is_batch=True) 
            # times+=1
            # val_accuracy,validation_accuracies,validation_epoches,train_loss_results,train_accuracy_results  \
                # = customized_train.trainings_val(model_cnn,input_images_train_norm2,input_images_test_norm2,y_train,y_test,LEARNING_RATE=LEARNING_RATE,BATCH_SIZE=BATCH_SIZE,num_epochs=EPOCH)
        
        # print (validation_accuracies)
        # print (validation_epoches)
    
    # accuracy,classesi = customized_train.test_accuacy(model_cnn,input_images_test_norm2,y_test)
    # print ("cnn" + str(testi) + '  {:0.4f}'.format(accuracy) )
    # print ("cnn" + str(testi) + '  {:4.2f}'.format(accuracy*100) )
    # accuracylist3.append (accuracy)
    # dat_out['predicted_cnn4'] = classesi
    # model_name = MODEL_DIR+base_name+'.model4.h5'
    # model_cnn.save(model_name) 
    
    #*****************************************************************************************************************
    ## random forest only run once 
    # if testi>=0:
        # continue 
    if True and layer_n==5:
    # if False and layer_n==4:
        clf = RandomForestClassifier(n_estimators=500) # 
        clf.fit(input_images_train2[:,:,:,:IMG_BANDS2].reshape(train_n,IMG_HEIGHT2*IMG_WIDTH2*IMG_BANDS2), y_train.reshape(train_n))
        classesi = clf.predict(input_images_test2[:,:,:,:IMG_BANDS2].reshape(test_n,IMG_HEIGHT2*IMG_WIDTH2*IMG_BANDS2))
        accuracy = (y_test.reshape(test_n)==classesi).sum()/classesi.size
        print ("rf" + str(testi) + '  {:0.4f}'.format(accuracy) )
        print ("rf" + str(testi) + '  {:4.2f}'.format(accuracy*100) )
        accuracylist3.append (accuracy)
        print (accuracy)
        dat_out['predicted_rf2' ] = classesi
    
    file_name = MODEL_DIR+base_name+".csv"
    dat_out.to_csv(file_name)
    
    print (accuracylist1)
    print (accuracylist2)
    print (accuracylist3)


#*****************************************************************************************************************
## partial CNN
# print ("\n\n#partial CNN *****************************************************************************************************************\n\n\n")
# accuracylist = list()
# accuracylist2 = list()
# importlib.reload(model_partial)
# for testi in range(ITERS):
    # print (testi+1)
    # model = model_partial.get_model_partial_cnn (IMG_HEIGHT=IMG_HEIGHT1,IMG_WIDTH=IMG_WIDTH1,IMG_BANDS=IMG_BANDS1,layer_n=layer_n,num_classes=N_CLASS, L2=L2,is_batch=True,is_dense_scale=True)    
    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # model.fit(input_images_train_norm1, y_train, validation_split=0.04, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=2)
    # logits = model.predict(input_images_test_norm1)
    # classesi = np.argmax(logits,axis=1).astype(np.uint8).reshape(y_test.shape)
    # accuracy = (y_test==classesi).sum()/classesi.size
    # print ("cnn" + str(testi) + '  {:0.4f}'.format(accuracy) )
    # accuracylist.append (accuracy)
    
