#!/usr/bin/env python3
__author__ = 'sibaji'
'''
This code reads nifty scans with 10 channels and generate GAD lesions segmentations.
This is training file
The 10 inputs are  five MRI contrasts (FLAIR, PD-weighted, T2-weighted, T1 pre-contrast, and T1 post-contrast) and 
five probability maps (gray matter, white matter, cerebrospinal fluid, lateral ventricles, and lesion probability maps).
The train_id3d1.txt, val_id3d1.txt and test_id3dallgad.txt file contains the gad lesion mask name with folder path
Same folder contains the MRI images with 5 cahnnels  and probability maps with 5 channels in nifty formats 

Sibaji Gaj
Lix6Lab
Example:
'''
######################################

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import Iterator
import numpy as np
import sys
import os
import nibabel as nib
import random
##########################################
img_ch=10 # image channels
out_ch=1 # output channel
img_rows=256
img_cols=256
batch_size=16
n_filters_g=32
val_ratio=0.2
init_lr=2e-4
nb_epoch=1000
alpha_recip=0.1
alpha_recip_cross=.3
alpha_recip_dice=1
alpha_recip_l1=1
model_out_dir='weights/'
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)

#################################################
def unet( n_filters=n_filters_g, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k=3 # kernel size
    s=2 # stride

    #img_height, img_width = img_size[0], img_size[1]
    padding='same'

    inputs = Input((img_rows, img_cols, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)


    
    return g
    
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-((2. * intersection + 1.e-10) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.e-10))
    
    
    
def bootstrapped_xentropy(y_true, y_pred, batch_size=batch_size, multiplier=12):
    
    number = 256 * multiplier
    loss=0
    #numLabels=y_pred.shape[0]
    for i in range(batch_size):
        y_true_flat = K.batch_flatten(y_true[i,...])
        y_pred_flat = K.batch_flatten(y_pred[i,...])
        batch_loss = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
        batch_loss=tf.contrib.framework.sort(batch_loss)
        batch_loss=batch_loss[-number:]
        loss+=K.mean(batch_loss)
    batch_size=K.cast(batch_size, dtype='float32')
    loss=loss/batch_size
    return loss
    

def get_data_single(image_id):

    imgs_mask=np.array(nib.load(image_id).dataobj) #reads the gad lesion masks
    name=image_id.replace("cg", "IM5")    
    imgs=np.array(nib.load(name).dataobj) #reads MRI scans 
    imgs=imgs-np.mean(imgs)
    imgs=imgs/np.std(imgs)
    name=name.split('.')[0]+'-PR.nii.gz'
    prob=np.array(nib.load(name).dataobj) #reads probablity scans 
    prob=prob/255.0
    imgs=np.append(imgs,prob,axis=-1)   
    imgs_mask=imgs_mask[...,np.newaxis]
    return imgs, imgs_mask


def predict_sigle_scan_single_instance(g, current_batch):
    pred_label=np.zeros((current_batch.shape[0],current_batch.shape[1],current_batch.shape[2],out_ch), dtype=np.float64)
    last_index=min(current_batch.shape[2],(int(current_batch.shape[2]/batch_size))*batch_size)
    for i in range(0,last_index,batch_size):
        real_imgs=np.zeros((batch_size,img_rows,img_cols,img_ch), dtype=np.float64)
        for j in range(batch_size):
            real_imgs[j,...]=current_batch[:,:,i+j,:]
        pred_mask=g.predict(real_imgs)
        for j in range(batch_size):
            pred_label[:,:,i+j,:]=pred_mask[j,...]

    if last_index != current_batch.shape[2]:
        for i in range(last_index,current_batch.shape[2]):
            real_imgs=current_batch[:,:,i,:]
            pred_mask=g.predict(real_imgs[np.newaxis,...])
            pred_label[:,:,i,:]=pred_mask


    return pred_label

def print_metrics(itr, **kargs):
    print ("*** Round {}  ====> ".format(itr),)
    for name, value in kargs.items():
        print (( "{} : {}, ".format(name, value)),end='')
    print ("")
    sys.stdout.flush()

###########################################
g = unet()
g.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=bootstrapped_xentropy, metrics=['accuracy']) # for bootstrapped_xentropy loss
#g.compile(optimizer = Adam(lr=init_lr, beta_1=0.5), loss = 'binary_crossentropy', metrics = ['accuracy']) # for cross entropy  loss
#g.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=dice_coef_loss, metrics=['accuracy']) # for dice  loss


g.summary()


train_id=[]
with open('train_id3d1.txt', 'r') as filehandle: #the text file contains the gad lesion mask name with location
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        train_id.append(currentPlace)

val_id=[]
with open('val_id3d1.txt', 'r') as filehandle: #the text file contains the gad lesion mask name with location
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        val_id.append(currentPlace)


print(len(train_id))
print(len(val_id))
best_acc=0
length_of_trainfiles_together=4 #read multiple 3D scans together for each iteration  
#train_id=train_id[:10]
#val_id=val_id[:10]

for n_round in range(nb_epoch+1):
    train_loss=[]
    np.random.shuffle(train_id)
    for current_batch_index in range(0,len(train_id),length_of_trainfiles_together):
        current_batch_list  = [train_id[i] for i in range(current_batch_index,min(len(train_id),current_batch_index+length_of_trainfiles_together))]
        current_batch_id=current_batch_list[0]
        current_batch2,current_label2=get_data_single(current_batch_id)
        for current_batch_id in current_batch_list[1:]:
            current_batch1, current_label1=get_data_single(current_batch_id)
            current_batch2=np.concatenate((current_batch2,current_batch1),axis=2)     #marge multiple scans together in slice directions
            current_label2=np.concatenate((current_label2,current_label1),axis=2)
        print(current_batch2.shape, current_label2.shape)
        for i in range(0,int((current_label2.shape[2]/batch_size)+0.5)): 
            imgs=np.zeros((batch_size,img_rows,img_cols,img_ch), dtype=np.float32)
            imgs_mask=np.zeros((batch_size,img_rows,img_cols,out_ch), dtype=np.float32)
            index=np.random.choice(current_label2.shape[2], batch_size, replace=False)
            for j in range(batch_size):                 #create batch by slecting random slices from multiple scans 
                imgs[j,...]=current_batch2[:,:,index[j],:]                   
                imgs_mask[j,...]=current_label2[:,:,index[j],:]
                
            if (np.sum(imgs_mask) > 2): #check if minimum GAD exist in the batch 
                loss, acc = g.train_on_batch(imgs, imgs_mask)
                print('loss',loss)
                train_loss.append(loss)
            else:                         #train for other no/less than minimum GAD for fewer times
                if random.randrange(200)==1:
                    loss, acc = g.train_on_batch(imgs, imgs_mask)
                    train_loss.append(loss)
    print_metrics(n_round, train_loss_sum=np.sum(train_loss), train_loss_mean=np.mean(train_loss))

        
        
    if n_round%1==0 : #goes for validation
        dice_coeff_total=[]
        for current_batch_index in range(0,len(val_id),1):
            current_batch_id  = [val_id[i] for i in range(current_batch_index,min(len(val_id),current_batch_index+1))]
            current_batch,current_label=get_data_single(current_batch_id[0]) #read single scans 
            pred_label=predict_sigle_scan_single_instance(g, current_batch)  # predict single scans
            pred_label[pred_label>.9]=1                                      # Thresold for dice calculation in validation scans  
            pred_label[pred_label<1]=0  
            dice_coeff1=(np.sum(pred_label*current_label)*2.0 +1.e-10)/ (np.sum(pred_label) + np.sum(current_label)+1.e-10) #calculate dice on validation scans
            dice_coeff_total.append(dice_coeff1)
        print_metrics(n_round, Val_dice_sum=np.sum(dice_coeff_total), Val_dice_mean=np.mean(dice_coeff_total) )   
        Val_dicefp_mean=(np.mean(dice_coeff_total))
        if n_round%50==0 :
            g.save_weights(os.path.join(model_out_dir,"g_{}_{:.3f}.h5".format(n_round,Val_dicefp_mean)))

            
        if (Val_dicefp_mean > best_acc):
            best_acc=Val_dicefp_mean
            if best_acc > 0:
                g.save_weights(os.path.join(model_out_dir,"g_{}_{:.3f}.h5".format(n_round,best_acc)))
                