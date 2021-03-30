#!/usr/bin/env python3
__author__ = 'sibaji'
'''
This code reads nifty scans with 10 channels and generate GAD lesions segmentations.
This is test file
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
save_folder_loc='pred/'
if not os.path.exists(save_folder_loc):
    os.makedirs(save_folder_loc)

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
#g.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=bootstrapped_xentropy, metrics=['accuracy']) # for bootstrapped_xentropy loss
#g.compile(optimizer = Adam(lr=init_lr, beta_1=0.5), loss = 'binary_crossentropy', metrics = ['accuracy']) # for cross entropy  loss
#g.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=dice_coef_loss, metrics=['accuracy']) # for dice  loss

g.load_weights('/home/gajs/osc/backup/Downloads/gald_seg/segmentation/Unet2D/Model1/weights8/g_445_0.702.h5')
g.summary()



test_id=[]
with open('test_id3dallgad.txt', 'r') as filehandle: #the text file contains the gad lesion mask name with location
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        test_id.append(currentPlace)


print(len(test_id))

#test_id=test_id[:10]
dice_coeff_total=[]
for current_batch_id in test_id:
    current_batch,current_label=get_data_single(current_batch_id) #read single scans 
    pred_label=predict_sigle_scan_single_instance(g, current_batch)  # predict single scans
    name=current_batch_id.split('/')[-1]
    name=name.replace(".nii", "pred.nii")
    nib.save(nib.Nifti1Image(np.asarray(pred_label, dtype=np.float32), nib.load(current_batch_id).get_affine()),save_folder_loc+'/'+   name )
    
        