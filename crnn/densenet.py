# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 17:05
# @Author  : xiongboying
# @Site    : 
# @File    : densenet.py
# @Software: PyCharm

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.layers import *

TRAIN = False

def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if (dropout_rate) and TRAIN:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate) and TRAIN:
        x = Dropout(dropout_rate)(x)

    if (pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif (pooltype == 1):
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif (pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter

def crnn2(input, nclass):
    m = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
    m = BatchNormalization(axis=3)(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(256, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
    m = BatchNormalization(axis=3)(m)

    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)

    m = Bidirectional(GRU(64, return_sequences=True, implementation=2), name='blstm1')(m)
    # m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(256, name='blstm1_out', activation='linear', )(m)
    # m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm2')(m)
    m = Bidirectional(GRU(64, return_sequences=True, implementation=2), name='blstm2')(m)
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()
    return y_pred

def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 32
    # conv 64 5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 64, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 64, _dropout_rate, 2, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x) #input 32 output 4*4*128
    x = Activation('relu')(x)
    x = AveragePooling2D((4, 1), strides=(4, 1))(x)

    x = Permute((2, 1, 3), name='permute')(x)

    # x = dense_blstm(x)  # gru

    x = TimeDistributed(Flatten(), name='flatten')(x)  # no-gru
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()

    return y_pred

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.
    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def mobile_conv_block(inputs, filters, kernel, strides, nl):
    """Convolution Block
    This function defines a 2D convolution operation with BN and activation.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)

def _squeeze(inputs):
    """Squeeze and Excitation.
    This function defines a squeeze structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
    """
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def bottleneck(inputs, filters, kernel, e, s, squeeze, nl):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        e: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        squeeze: Boolean, Whether to use the squeeze.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = K.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(1.0 * filters)

    r = s == 1 and input_shape[3] == filters

    x = mobile_conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)

    if squeeze:
        x = _squeeze(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x

def MobileNetV3_Cnn(input, nclass):

    x = mobile_conv_block(input, 128, (3, 3), strides=(2, 2), nl='HS')
    x = bottleneck(x, 128, (3, 3), e=16, s=2, squeeze=True, nl='RE')
    x = bottleneck(x, 128, (3, 3), e=72, s=2, squeeze=False, nl='RE')

    # x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)  # input 32 output 4*4*128
    # x = Activation('relu')(x)
    x = AveragePooling2D((4, 1), strides=(4, 1))(x)

    x = Permute((2, 1, 3), name='permute')(x)

    x = TimeDistributed(Flatten(), name='flatten')(x)  # no-gru
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred

def dense_blstm(input):
    rnn_size = 128

    conv_shape = input.get_shape()
    x = Reshape(target_shape=(-1, int(conv_shape[2] * conv_shape[3])))(input)
    x = Dense(32, activation='relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
    gru1_merged = merge.Add()([gru_1, gru_1b])

    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
    x = merge.Concatenate()([gru_2, gru_2b])
    if TRAIN:
        x = Dropout(0.25)(x)

    return x
# input = Input(shape=(32, 280, 1), name='the_input')
# dense_cnn(input, 5000)
