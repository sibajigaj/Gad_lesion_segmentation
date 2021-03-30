#!/usr/bin/env python3
__author__ = 'sibaji'
'''
This code reads nifty scans with 5 channels and generate GAD lesions segmentations.
This is testing file. Generates the predicted lesion mask from input scans 
The 5 chnnels  are  five MRI contrasts (T1 post-contrast, T1 pre-contrast, FLAIR,  T2-weighted and PD-weighted  ).
The train_id3d1.txt, val_id3d1.txt and test_id3dallgad.txt file contains the gad lesion mask name with folder path
Same folder contains the MRI images with 5 cahnnels  in nifty formats 
Sibaji Gaj
Lix6Lab
Example:
'''
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D,Reshape
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
#from keras_contrib.layers.normalization import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
K.set_image_data_format('channels_last') 
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate
import nibabel as nib
import os
import sys
import random
from keras import objectives
import random

#############################
initial_learning_rate=0.00001
img_rows=256   
img_cols=256
img_hight=8
img_ch=5
batch_size=1
n_labels=1
input_file_path='/users/PCCF0019/ccf0064/deep_learning_segment_gad/'
model_out_dir='weights1/'
if not os.path.exists(model_out_dir):
    os.mkdir(model_out_dir)
nb_epoch=200
############################
def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=n_labels,  deconvolution=False,
                  depth=3, n_base_filters=32, include_label_wise_dice_coefficients=False, 
                  batch_normalization=True, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    
    
    # set image specifics
    kernel=(3, 3, 3) # kernel size
    s=2 # stride

    #img_height, img_width = img_size[0], img_size[1]
    padding='same'

    conv1 = create_convolution_block(input_layer=inputs, n_filters=n_base_filters*4, # 256 * 256 *8
                                          batch_normalization=batch_normalization)
    conv1 = create_convolution_block(input_layer=conv1, n_filters=n_base_filters*4,
                                          batch_normalization=batch_normalization)
    pool1 = Conv3D(n_base_filters, (3, 3, 3), padding=padding, strides=(2,2,1))(conv1)
    
    conv2 = create_convolution_block(input_layer=pool1, n_filters=n_base_filters*4,       # 128 * 128 *8
                                          batch_normalization=batch_normalization)
    conv2 = create_convolution_block(input_layer=conv2, n_filters=n_base_filters*4,
                                          batch_normalization=batch_normalization)
    pool2 = Conv3D(n_base_filters*2, (3, 3, 3), padding=padding, strides=(1,1,2))(conv2)
    
    conv3 = create_convolution_block(input_layer=pool2, n_filters=n_base_filters*2,         # 128 * 128 *4
                                          batch_normalization=batch_normalization)
    conv3 = create_convolution_block(input_layer=conv3, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    pool3 = Conv3D(n_base_filters*4, (3, 3, 3), padding=padding, strides=(2,2,1))(conv3)
    
    conv4 = create_convolution_block(input_layer=pool3, n_filters=n_base_filters*2,         # 64 * 64 *4
                                          batch_normalization=batch_normalization)
    conv4 = create_convolution_block(input_layer=conv4, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    pool4 = Conv3D(n_base_filters*2, (3, 3, 3), padding=padding, strides=(2,2,1))(conv4)
    
    conv5 = create_convolution_block(input_layer=pool4, n_filters=n_base_filters,         # 32 * 32 *4
                                          batch_normalization=batch_normalization)
    conv5 = create_convolution_block(input_layer=conv5, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    pool5 = Conv3D(n_base_filters*4, (3, 3, 3), padding=padding, strides=(2,2,2))(conv5)
    
    conv6 = create_convolution_block(input_layer=pool5, n_filters=n_base_filters*2,         # 16 * 16 *2
                                          batch_normalization=batch_normalization)
    conv6 = create_convolution_block(input_layer=conv6, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
                                          
    
    
    convskip0=Conv3D(n_base_filters*8, (1, 1, 1), padding=padding, strides=(1,1,1))(inputs)
    convskip0 = BatchNormalization(axis=-1)(convskip0)
    convskip0=Activation('relu')(convskip0)
    
    convskip0=Conv3D(n_base_filters*4, (2, 2, 1), padding=padding, strides=(1,1,1))(convskip0)
    convskip0 = BatchNormalization(axis=-1)(convskip0)
    convskip0=Activation('relu')(convskip0)
    
    convskip0=Conv3D(n_base_filters*4, (2, 2, 1), padding=padding, strides=(1,1,1))(convskip0)
    convskip0 = BatchNormalization(axis=-1)(convskip0)
    convskip0=Activation('relu')(convskip0)
    
    convskip0=Conv3D(n_base_filters*4, (1, 1, 2), padding=padding, strides=(1,1,1))(convskip0)
    convskip0 = BatchNormalization(axis=-1)(convskip0)
    convskip0=Activation('relu')(convskip0)
    
    convskip1=Conv3D(n_base_filters*4, (1, 1, 1), padding=padding, strides=(1,1,1))(conv1)
    convskip1 = BatchNormalization(axis=-1)(convskip1)
    convskip1=Activation('relu')(convskip1)
    
    convskip1=Conv3D(n_base_filters*4, (1, 1, 2), padding=padding, strides=(1,1,1))(convskip1)
    convskip1 = BatchNormalization(axis=-1)(convskip1)
    convskip1=Activation('relu')(convskip1)
    
    convskip1=Conv3D(n_base_filters*4, (1, 1, 1), padding=padding, strides=(1,1,1))(convskip1)
    convskip1 = BatchNormalization(axis=-1)(convskip1)
    convskip1=Activation('relu')(convskip1)
    
    convskip1=Conv3D(n_base_filters*4, (1, 1, 2), padding=padding, strides=(1,1,1))(convskip1)
    convskip1 = BatchNormalization(axis=-1)(convskip1)
    convskip1=Activation('relu')(convskip1)
    
    
    convskip2=Conv3D(n_base_filters*2, (1, 1, 1), padding=padding, strides=(1,1,1))(conv2)
    convskip2 = BatchNormalization(axis=-1)(convskip2)
    convskip2=Activation('relu')(convskip2)
    
    convskip2=Conv3D(n_base_filters*2, (1, 1, 2), padding=padding, strides=(1,1,1))(convskip2)
    convskip2 = BatchNormalization(axis=-1)(convskip2)
    convskip2=Activation('relu')(convskip2)
    
    convskip2=Conv3D(n_base_filters*2, (1, 1, 1), padding=padding, strides=(1,1,1))(convskip2)
    convskip2 = BatchNormalization(axis=-1)(convskip2)
    convskip2=Activation('relu')(convskip2)
    
    convskip2=Conv3D(n_base_filters*2, (1, 1, 2), padding=padding, strides=(1,1,1))(convskip2)
    convskip2 = BatchNormalization(axis=-1)(convskip2)
    convskip2=Activation('relu')(convskip2)
    
    
    up_convolution7=UpSampling3D(size=(2, 2, 2))(conv6)                                      # 32 * 32 *4
    concat7 = concatenate([up_convolution7, conv5], axis=-1)
    conv7 = create_convolution_block(input_layer=concat7, n_filters=n_base_filters*2,         
                                          batch_normalization=batch_normalization)
    conv7 = create_convolution_block(input_layer=conv7, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
                                          
    up_convolution8=UpSampling3D(size=(2, 2, 1))(conv7)                                     # 64 * 64 *4                                  
    concat8 = concatenate([up_convolution8, conv4], axis=-1)
    conv8 = create_convolution_block(input_layer=concat8, n_filters=n_base_filters,        
                                          batch_normalization=batch_normalization)
    conv8 = create_convolution_block(input_layer=conv8, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
                                          
    up_convolution8=UpSampling3D(size=(2,2,1))(conv8)                                     # 128 * 128 *4                                  
    concat9 = concatenate([up_convolution8, conv3], axis=-1)
    conv9 = create_convolution_block(input_layer=concat9, n_filters=n_base_filters,        
                                          batch_normalization=batch_normalization)
    conv9 = create_convolution_block(input_layer=conv9, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)  
                                          
    up_convolution10=UpSampling3D(size=(1,1,2))(conv9)                                     # 128 * 128 *8                                 
    concat10 = concatenate([up_convolution10, conv2, convskip2], axis=-1)
    conv10 = create_convolution_block(input_layer=concat10, n_filters=n_base_filters,        
                                          batch_normalization=batch_normalization)
    conv10 = create_convolution_block(input_layer=conv10, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)                                       
                                          
    
    up_convolution11=UpSampling3D(size=(2,2,1))(conv10)                                     # 256*256 *8                                 
    concat11 = concatenate([up_convolution11, conv1, convskip0], axis=-1)
    conv11 = create_convolution_block(input_layer=concat11, n_filters=n_base_filters*4,        
                                          batch_normalization=batch_normalization)
    conv11 = create_convolution_block(input_layer=conv11, n_filters=n_base_filters*4,
                                          batch_normalization=batch_normalization)
    
    
    concat12 = concatenate([conv11, convskip1], axis=-1)
    concat12 = create_convolution_block(input_layer=conv11, n_filters=n_base_filters*4,        
                                          batch_normalization=batch_normalization)
    concat12 = create_convolution_block(input_layer=concat12, n_filters=n_base_filters*4,
                                          batch_normalization=batch_normalization)
                                          
                                      
    

    final_convolution = Conv3D(n_labels, (1, 1, 1))(conv11)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    
    return model

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=True):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
        
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-((2. * intersection + 1.e-10) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.e-10))
    
    

def get_data_single3d(image_id):
    
    imgs_mask=np.array(nib.load(image_id).dataobj)
    image_id1=image_id.replace('cg','IM5')
    imgs=np.array(nib.load(image_id1).dataobj)

    imgs_mask=imgs_mask[...,np.newaxis]
    return imgs[np.newaxis,...], imgs_mask[np.newaxis,...]

def predict_single_image(real_imgs):
    pred_mask=np.zeros((real_imgs.shape[1],real_imgs.shape[2],real_imgs.shape[3]))
    for k in range(0,256-img_cols+1,img_cols):
        for j in range(0,256-img_cols+1,img_cols):
            for i in range(0,65-img_hight,img_hight):
                imgs=real_imgs[:,k:k+img_rows,j:j+img_cols,i:i+img_hight,:]
                pred_mask1=g.predict(imgs)
                pred_mask1=np.squeeze(pred_mask1)
                #print(pred_mask.shape, pred_mask1.shape)
                pred_mask[k:k+img_rows,j:j+img_cols,i:i+img_hight]=pred_mask1[...]
    imgs=real_imgs[:,-img_rows:,-img_cols:,-img_hight:,:]
    #print(imgs.shape)
    pred_mask1=g.predict(imgs)
    pred_mask[-img_rows:,-img_cols:,-img_hight:]=np.squeeze(pred_mask1)

    return pred_mask


##########################################################################

g=unet_model_3d((img_rows,img_cols,img_hight,img_ch),depth=5, n_base_filters=32,  batch_normalization=True)


weight_name=sys.argv[1]  #provide the weights to use 
save_folder_loc=sys.argv[2] #provide the location to save the preictions 

g.load_weights(weight_name)

print('Running weight:',weight_name,'save_folder_loc',save_folder_loc)

###############################################################################################
save_folder_loc1=save_folder_loc+'test/GAD/'
if not os.path.exists(save_folder_loc1):
        os.makedirs(save_folder_loc1)

test_id=[]
with open('test_id3dallgad.txt', 'r') as filehandle: #the text file contains the gad lesion mask name with location
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        test_id.append(currentPlace)

print(len(test_id))

for current_batch_id in test_id:
    current_batch,current_label=get_data_single3d(current_batch_id)
    
    pred_label=predict_single_image(current_batch)

    
    pred_label=np.squeeze(pred_label)
    current_label=np.squeeze(current_label)
    pred_label[pred_label>.9]=1
    pred_label[pred_label<1]=0
    dice_coeff1=(np.sum(pred_label*current_label)*2.0 +1.e-10)/ (np.sum(pred_label) + np.sum(current_label)+1.e-10)
    #dice_coeff_total.append(dice_coeff1)
    print(current_batch_id,dice_coeff1)


    name=current_batch_id.split('/')[-1]
    name=name.replace(".nii", "pred.nii")



    nib.save(nib.Nifti1Image(np.asarray(pred_label, dtype=np.float32), nib.load(current_batch_id).get_affine()),save_folder_loc1+'/'+   name )
