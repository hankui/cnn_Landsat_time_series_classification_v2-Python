# import train_test 
import os 
import math 
import numpy as np  
import pandas as pd 


SPLIT_DIR = "./split/"

if not os.path.isdir(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

train_fields1 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    # for peri in ('low10', 'low25', 'middle','high75', 'high90'):
    for peri in ('low25', 'middle','high75'):
        # print(bandi)
        train_fields1.append(peri+'.'+bandi)

train_fields2 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low25', 'middle','high75', 'high90'):
    # for peri in ('low25', 'middle','high75'):
        # print(bandi)
        train_fields2.append(peri+'.'+bandi)

train_fields7 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low20', 'low35', 'middle','high65', 'high80', 'high90'):
    # for peri in ('low10', 'low35', 'middle','high65', 'high90'):
        train_fields7.append(peri+'.'+bandi)

train_fields9 = list()
for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi','ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5', 'ratio6', 'ratio7'):
    for peri in ('low10', 'low20', 'low30', 'low40', 'middle','high60', 'high70', 'high80', 'high90'):
        train_fields9.append(peri+'.'+bandi)

train_fields_com = list()
for comi in range(14):
    peri='N{:02d}'.format(comi)
    for bandi in ('green', 'red', 'nir','swir1', 'swir2','ndvi'):
        train_fields_com.append(peri+'.'+bandi)

def get_training_test_com(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.1):
    if proportion==0.5:
        index_train = np.logical_and(orders<5,valid_index)
        index_test  = np.logical_and(orders>=5,valid_index)
    elif proportion==0.9:
        index_train = np.logical_and(orders<9,valid_index)
        index_test  = np.logical_and(orders>=9,valid_index)
    else:
        index_train = np.logical_and(orders==0,valid_index)
        index_test  = np.logical_and(orders!=0,valid_index)
    
    train_fieldsx = train_fields_com
    # input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train2,std_train2 \
        # = construct_metric_train_test\
        # (data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2)
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2)
    
    
    dat_out = pd.DataFrame()
    for ai in ('hh', 'vv', 'col', 'row', 'nlcd'):
        dat_out[ai] = (data_per[ai][index_test]).copy()
    
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    dat_out['predicted_rf1' ] = 255
    dat_out['predicted_rf2' ] = 255
    return y_train,y_test,\
        input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train        

def get_training_test7(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.1):
    if proportion==0.5:
        index_train = np.logical_and(orders<5,valid_index)
        index_test  = np.logical_and(orders>=9,valid_index)
    elif proportion==0.9:
        index_train = np.logical_and(orders<9,valid_index)
        index_test  = np.logical_and(orders>=9,valid_index)
    else:
        index_train = np.logical_and(orders==0,valid_index)
        index_test  = np.logical_and(orders>=9,valid_index)
    
    # IMG_HEIGHT2 = 3   ; IMG_WIDTH2=13; IMG_BANDS2=1
    train_fieldsx = train_fields7
    if IMG_HEIGHT2 == 7:
        train_fieldsx = train_fields7
    elif IMG_HEIGHT2 == 9:
        train_fieldsx = train_fields9
    elif IMG_HEIGHT2 == 5:
        train_fieldsx = train_fields2
    elif IMG_HEIGHT2 == 3:
        train_fieldsx = train_fields1
    else:
        train_fieldsx = train_fields_com
        
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train2,std_train2 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2)
    
    # for 1d cnn
    input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,y_train,y_test,mean_train3,std_train3 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fieldsx,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2)
    
    dat_out = pd.DataFrame()
    for ai in ('hh', 'vv', 'col', 'row', 'nlcd'):
        dat_out[ai] = (data_per[ai][index_test]).copy()
    
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    dat_out['predicted_rf1' ] = 255
    dat_out['predicted_rf2' ] = 255
    return y_train,y_test,\
        input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,\
        input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,dat_out, mean_train2,std_train2        


def get_training_test(data_per,orders,valid_index,IMG_HEIGHT1,IMG_WIDTH1,IMG_BANDS1,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.1):
    if proportion==0.5:
        index_train = np.logical_and(orders<5,valid_index)
        index_test =  np.logical_and(orders>=5,valid_index)
    elif proportion==0.9:
        index_train = np.logical_and(orders<9,valid_index)
        index_test =  np.logical_and(orders>=9,valid_index)
    else:
        index_train = np.logical_and(orders==0,valid_index)
        index_test =  np.logical_and(orders!=0,valid_index)
    
    # importlib.reload(train_test)
    ## IMG_HEIGHT2 = 5   ; IMG_WIDTH2=13; IMG_BANDS2=1
    input_images_train_norm1,input_images_test_norm1,input_images_train1,input_images_test1,y_train,y_test,mean_train1,std_train1 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fields1,class_field,IMG_HEIGHT=IMG_HEIGHT1,IMG_WIDTH=IMG_WIDTH1,IMG_BANDS=IMG_BANDS1)
    
    # IMG_HEIGHT2 = 3   ; IMG_WIDTH2=13; IMG_BANDS2=1
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train2,std_train2 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fields2,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2,IMG_BANDS=IMG_BANDS2)
    
    # for 1d cnn
    input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,y_train,y_test,mean_train3,std_train3 \
        = construct_metric_train_test\
        (data_per,index_train,index_test,train_fields2,class_field,IMG_HEIGHT=IMG_HEIGHT2,IMG_WIDTH=IMG_WIDTH2)
    
    dat_out = pd.DataFrame()
    for ai in ('hh', 'vv', 'col', 'row', 'nlcd'):
        dat_out[ai] = (data_per[ai][index_test]).copy()
    
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    dat_out['predicted_rf1' ] = 255
    dat_out['predicted_rf2' ] = 255
    return input_images_train_norm1,input_images_test_norm1,input_images_train1,input_images_test1,y_train,y_test,\
        input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,\
        input_images_train_norm3,input_images_test_norm3,input_images_train3,input_images_test3,dat_out, mean_train2,std_train2        

def random_split_train_validation (X_train,y_train,pecentage = 0.04):
    total_n = y_train.shape[0]
    sample_n = math.ceil(total_n*pecentage)
    split_n  = math.ceil(total_n/sample_n)
    file_index = SPLIT_DIR+"split.total_n"+str(total_n)+".for.validation.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    validation_index = orders==0
    training_index   = orders!=0
    sum(validation_index)
    sum(training_index  )
    return X_train[training_index],y_train[training_index],X_train[validation_index],y_train[validation_index],training_index,validation_index


def random_split (total_n, split_n):
    sample_n = math.ceil(total_n/split_n)
    file_index = SPLIT_DIR+"index.total_n"+str(total_n)+".for.random.txt"
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    return orders



# IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
# IMG_HEIGHT = 5   ; IMG_WIDTH=14; IMG_BANDS=1
def construct_metric_train_test(data_per,index_train,index_test,train_fields,test_field,\
    IMG_HEIGHT,IMG_WIDTH,IMG_BANDS=0):
    
    # test_field = 
    
    trainx2 = np.array(data_per[train_fields][index_train]).astype(np.float32)
    trainx2[np.isnan(trainx2)] = 0 
    trainx2[np.isinf(trainx2)] = 0 
    if IMG_BANDS==0:
        input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT)
    else:
        input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    
    y_train = np.array(data_per[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_per[train_fields][index_test]).astype(np.float32)
    testx2[np.isnan(testx2)] = 0 
    testx2[np.isinf(testx2)] = 0 
    if IMG_BANDS==0:
        input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT)
    else:
        input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    
    y_test = np.array(data_per[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(train_n)
    print(test_n )
    print(np.isnan(input_images_train).sum()/input_images_train.size*100)
    print(np.isnan(input_images_test ).sum()/input_images_test .size*100)
    
    ## check data
    # for i in range(input_images_train.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_train[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    # for i in range(input_images_test.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_test[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    
    # input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    # masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    # input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    # input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0
    # masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    # input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    # input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train.copy()
    input_images_test_norm0  = input_images_test .copy()
    
    ## this norm turn out to be very important 
    # a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
        # mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    
    if IMG_BANDS==0:
        a = np.concatenate((input_images_train[:,:,:],input_images_test[:,:,:]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2])
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2])
        input_images_train_norm0[:,:,:] = (input_images_train[:,:,:] - mean_train)/std_train
        input_images_test_norm0 [:,:,:] = (input_images_test [:,:,:] - mean_train)/std_train
    else:
        a = np.concatenate((input_images_train[:,:,:,0],input_images_test[:,:,:,0]))
        mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
        std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
        input_images_train_norm0[:,:,:,:IMG_BANDS] = (input_images_train[:,:,:,:IMG_BANDS] - mean_train)/std_train
        input_images_test_norm0 [:,:,:,:IMG_BANDS] = (input_images_test [:,:,:,:IMG_BANDS] - mean_train)/std_train
    # b = np.ma.array(input_images_train_norm0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # b.mean(axis=0)
    # b.std(axis=0)
    return input_images_train_norm0,input_images_test_norm0,input_images_train,input_images_test,y_train,y_test,mean_train,std_train


# IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
def construct_composite_train_test(data_all,index_train,index_test,train_fields,test_field,\
    IMG_HEIGHT,IMG_WIDTH,IMG_BANDS):
    
    # test_field = 
    
    trainx2 = np.array(data_all[train_fields][index_train]).astype(np.float32)
    input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_train = np.array(data_all[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_all[train_fields][index_test]).astype(np.float32)
    input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_test = np.array(data_all[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(train_n)
    print(test_n )
    print(np.isnan(input_images_train).sum()/input_images_train.size*100)
    print(np.isnan(input_images_test ).sum()/input_images_test .size*100)
    
    ## check data
    # for i in range(input_images_train.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_train[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    # for i in range(input_images_test.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_test[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0
    masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train0.copy()
    input_images_test_norm0  = input_images_test0 .copy()
    
    ## this norm turn out to be very important 
    a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
        mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    
    mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
    std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
    # mean_train = 0
    # std_train  = 1
    input_images_train_norm0[:,:,:,:IMG_BANDS] = (input_images_train0[:,:,:,:IMG_BANDS] - mean_train)/std_train
    input_images_test_norm0 [:,:,:,:IMG_BANDS] = (input_images_test0 [:,:,:,:IMG_BANDS] - mean_train)/std_train
    # b = np.ma.array(input_images_train_norm0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # b.mean(axis=0)
    # b.std(axis=0)
    return input_images_train_norm0,input_images_test_norm0,input_images_train0,input_images_test0,y_train,y_test,mean_train,std_train
