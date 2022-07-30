"""
-------------------->  Different Models architecture implementation  <--------------------
Current Models:
    - UNet 1 --> Initial Unet for Lung Segmentation Architecture
    - UNet 2 --> Light Unet Architecture
    - UNet 3 --> Standard Unet Architecture
    - UNetPlus 1 --> Not Implemented
    - UNetPlusPlus 1 --> Original UNetPlusPlus architecture
    - UNetPlusPlus 2 --> Enhanced UNetPlusPlus architecture (Not Implemented)

Notes: Unit.py is highly correlated with this file
"""

import numpy as np


import tensorflow as tf

import keras
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.models import Model
from Unit import standard_unit

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K


'''
Hyper-parameters
'''
# input data
INPUT_SIZE = 128
INPUT_CHANNEL = 1   # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 1
# network structure
FILTER_NUM = 32 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 2 # size of upsampling filters

'''
Definitions of loss and evaluation metrices
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    axis = 3
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=axis)(shortcut)

    res_path = tf.keras.layers.add([shortcut, conv])
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = tf.keras.layers.add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = tf.keras.layers.multiply([upsample_psi, x])

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn


def Attention_ResUNet(dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet construction, with attention gate
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: model
    '''
    # input data
    # dimension of the image depth
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL), dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = Conv2D(OUTPUT_MASK_CHANNEL, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('relu', name = 'attention_vec')(conv_final)

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    return model






def UNet1(img_rows=256, img_cols=256, color_type=1, num_class=1, multiclass=True):
    """ 
        ---------> Network Info <---------
        
        Initial Unet for Lung Segmentation Architecture
        
        Total params: 31,043,465
        Trainable params: 31,037,575
        Non-trainable params: 5,890

        Comments:
        The accuracy of this architecture is quiet high but the problem is its high number of parameters which result
        in slow learning process, memory shortage and etc.
        It is also interesting to mention that with this model we can not train all the data in one attemp (lack of memory) and we need
        to train the model iteratively

        Credits:
        Prof. Catalin Fetita and Ali Keshavarzi
    """

    inputs = Input((img_rows, img_cols, 1))
    BN0 = BatchNormalization()(inputs)
    
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    BN1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(BN1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    BN2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(BN2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    BN3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(BN3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    BN4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(BN4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    BN5 = BatchNormalization()(conv5)
    encode = [BN1, BN2, BN3, BN4, BN5]
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(BN5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    BN6 = BatchNormalization()(up6)
    merge6 = concatenate([encode[-2], BN6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    BN7 = BatchNormalization()(up7)
    merge7 = concatenate([encode[-3], BN7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    BN8 = BatchNormalization()(up8)
    merge8 = concatenate([encode[-4], BN8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    BN9 = BatchNormalization()(up9)
    merge9 = concatenate([encode[-5], BN9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    if multiclass:
        print("Multi-Class Segmentation...")
        conv10 = Conv2D(1, 1, activation='softmax')(conv9)
    else:
        print("Single-Class Segmentation...")
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    return model

def UNet2(img_rows=256, img_cols=256, color_type=1, num_class=1, multiclass=True):
    
    """ 
        ---------> Network Info <---------
        
        Light Unet Architecture

        Total params: 1,940,817
        Trainable params: 1,940,817
        Non-trainable params: 0

        Comments:
        ... ...

        Credits:
        Zhou et al. (I slightly changed (mostly last layer) the architecture)
        UNet++: A Nested U-Net Architecture for Medical Image Segmentation, Zhou et.al, 2018
        Link: https://arxiv.org/abs/1807.10165
        Github repo of the original paper: https://github.com/MrGiovanni/UNetPlusPlus

    """
    inputs = tf.keras.layers.Input((img_rows, img_cols, 1))

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    if (multiclass):
        print("Multi-Class Segmentation...")
        outputs = tf.keras.layers.Conv2D(num_class, (1, 1), activation='softmax')(c9)
    else:
        print("Single-Class Segmentation...")
        outputs = tf.keras.layers.Conv2D(num_class, (1, 1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model

def UNet3(img_rows=256, img_cols=256, color_type=1, num_class=1, multiclass=True):
   
    """ 
        ---------> Network Info <---------
        
        Standard Unet Architecture
        
        Total params: 7,759,521
        Trainable params: 7,759,521
        Non-trainable params: 0

        Comments:
        ... ...

        Credits:
        U-Net: Convolutional Networks for Biomedical Image Segmentation, Ronneberger et.al, 2015 
        Link: https://arxiv.org/abs/1505.04597

    """

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
    
    if (multiclass):
        print("Multi-Class Segmentation...")
        unet_output = Conv2D(num_class, (1, 1), activation='softmax', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    else:
        print("Single-Class Segmentation...")
        unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    
    model = Model(img_input, unet_output)

    return model

def UNetTest(img_rows=128, img_cols=128, color_type=1, num_class=1, multiclass=True):
    inputs = tf.keras.Input((img_rows, img_cols, 1))
    x = inputs
    depth = 4
    features = 16
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    for i in reversed(range(depth)):
        features = features // 2
        # attention_up_and_concate(x,[skips[i])
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([skips[i], x // 2], axis=1)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same')(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model



def UNetPlus1(img_rows=256, img_cols=256, color_type=1, num_class=1):
    """ 
        ---------> Network Info <---------
        
        Wide UNet Architecture
        
        Total params: -
        Trainable params: -
        Non-trainable params: -
        
        Comments:
        ... ...

        Credits:

    """
    # Todo - Implementing UNetPlus Architecture Here !
    return 

def UNetPlusPlus1(img_rows=128, img_cols=128, color_type=1, num_class=1, deep_supervision=False, multiclass=True):
    """ 
        ---------> Network Info <---------
        
        Initial UNetPlusPlus Architecture
        
        Total params: 9,041,601
        Trainable params: 9,041,601
        Non-trainable params: 0
        
        Comments:
        ... ...

        Credits:
        Zhou et al. (I slightly changed (mostly last layer) the architecture)
        UNet++: A Nested U-Net Architecture for Medical Image Segmentation, Zhou et.al, 2018
        Link: https://arxiv.org/abs/1807.10165
        Github repo of the original paper: https://github.com/MrGiovanni/UNetPlusPlus

    """   
    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    if multiclass:
        print("Multi-Class Segmentation...")
        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='softmax', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='softmax', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='softmax', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='softmax', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    else:
        print("Single-Class Segmentation...")
        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4])
    else:
        model = Model(img_input, [nestnet_output_4])
        
    return model


def UNet1(img_rows=256, img_cols=256, color_type=1, num_class=1, multiclass=False):
    merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(filter_num * 4, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # res0 = residual_block_2d(inputs, output_channels=filter_num * 2)

    pool = MaxPooling2D(pool_size=(2, 2))(conv1)

    res1 = residual_block_2d(pool, output_channels=filter_num * 4)

    # res1 = residual_block_2d(atb1, output_channels=filter_num * 4)

    pool1 = MaxPooling2D(pool_size=(2, 2))(res1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(atb1)

    res2 = residual_block_2d(pool1, output_channels=filter_num * 8)

    # res2 = residual_block_2d(atb2, output_channels=filter_num * 8)
    pool2 = MaxPooling2D(pool_size=(2, 2))(res2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(atb2)

    res3 = residual_block_2d(pool2, output_channels=filter_num * 16)
    # res3 = residual_block_2d(atb3, output_channels=filter_num * 16)
    pool3 = MaxPooling2D(pool_size=(2, 2))(res3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(atb3)

    res4 = residual_block_2d(pool3, output_channels=filter_num * 32)

    # res4 = residual_block_2d(atb4, output_channels=filter_num * 32)
    pool4 = MaxPooling2D(pool_size=(2, 2))(res4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(atb4)

    res5 = residual_block_2d(pool4, output_channels=filter_num * 64)
    # res5 = residual_block_2d(res5, output_channels=filter_num * 64)
    res5 = residual_block_2d(res5, output_channels=filter_num * 64)

    atb5 = attention_block_2d(res4, encoder_depth=1, name='atten1')
    up1 = UpSampling2D(size=(2, 2))(res5)
    merged1 = concatenate([up1, atb5], axis=merge_axis)
    # merged1 = concatenate([up1, atb4], axis=merge_axis)

    res5 = residual_block_2d(merged1, output_channels=filter_num * 32)
    # atb5 = attention_block_2d(res5, encoder_depth=1)

    atb6 = attention_block_2d(res3, encoder_depth=2, name='atten2')
    up2 = UpSampling2D(size=(2, 2))(res5)
    # up2 = UpSampling2D(size=(2, 2))(atb5)
    merged2 = concatenate([up2, atb6], axis=merge_axis)
    # merged2 = concatenate([up2, atb3], axis=merge_axis)

    res6 = residual_block_2d(merged2, output_channels=filter_num * 16)
    # atb6 = attention_block_2d(res6, encoder_depth=2)

    # atb6 = attention_block_2d(res6, encoder_depth=2)
    atb7 = attention_block_2d(res2, encoder_depth=3, name='atten3')
    up3 = UpSampling2D(size=(2, 2))(res6)
    # up3 = UpSampling2D(size=(2, 2))(atb6)
    merged3 = concatenate([up3, atb7], axis=merge_axis)
    # merged3 = concatenate([up3, atb2], axis=merge_axis)

    res7 = residual_block_2d(merged3, output_channels=filter_num * 8)
    # atb7 = attention_block_2d(res7, encoder_depth=3)

    # atb7 = attention_block_2d(res7, encoder_depth=3)
    atb8 = attention_block_2d(res1, encoder_depth=4, name='atten4')
    up4 = UpSampling2D(size=(2, 2))(res7)
    # up4 = UpSampling2D(size=(2, 2))(atb7)
    merged4 = concatenate([up4, atb8], axis=merge_axis)
    # merged4 = concatenate([up4, atb1], axis=merge_axis)

    res8 = residual_block_2d(merged4, output_channels=filter_num * 4)
    # atb8 = attention_block_2d(res8, encoder_depth=4)

    # atb8 = attention_block_2d(res8, encoder_depth=4)
    up = UpSampling2D(size=(2, 2))(res8)
    # up = UpSampling2D(size=(2, 2))(atb8)
    merged = concatenate([up, conv1], axis=merge_axis)
    # res9 = residual_block_2d(merged, output_channels=filter_num * 2)

    conv9 = Conv2D(filter_num * 4, 3, padding='same')(merged)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    output = Conv2D(1, 3, padding='same', activation='sigmoid')(conv9)
    model = Model(inputs, output)
    return model



if __name__ == '__main__':
    
    model = UNet1(96,96,1)
    model.summary()

    model = UNet2(96,96,1)
    model.summary()

    model = UNet3(96,96,1)
    model.summary()

    model = UNetPlus1(96,96,1)
    model.summary()

    model = UNetPlusPlus1(96,96,1)
    model.summary()
    
    model = UNetTest(96,96,1)
    model.summary()
