# -*- coding:utf-8 -*-
import os
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from imp import reload

# from auto_generator import AutoImageDataGenerator
from file_generator import FileImageDataGenerator
import densenet

CHAR_FILE_PATH = '../ext_data/corpus/char(120w).txt'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/name_weights'

DATA_PATH = 'D:/projects/data/ccf2019_ocr/name_l2'

# 训练权重日期版本（不要重复）
DATE_VER = '1013'+ '_name'

image_resize_times = 2
img_h = int(40 * image_resize_times)
img_w = int(110 * image_resize_times)

batch_size = 512
maxlabellength = 4

chardic = {}
numdic ={}
charfile = open(CHAR_FILE_PATH, encoding='utf8')
lines = charfile.readlines()
char_set = open(CHAR_FILE_PATH, 'r', encoding='utf-8').readlines()
char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
nclass = len(char_set)

for line in lines:
    line = line.replace('\n','')
    #key是字符
    chardic[line] = lines.index(line+'\n')

for line in lines:
    line = line.replace('\n','')
    numdic[lines.index(line+'\n')] = line

def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    gpu_options.allow_growth = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # session = tf.Session(config=config,)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        # return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        return tf.Session(config=config)
#
# def readfile(filename,encoding='utf-8'):
#     res = []
#     with open(filename, 'r', encoding=encoding) as f:
#         lines = f.readlines()
#         for i in lines:
#             res.append(i.strip())
#     dic = {}
#     for i in res:
#         p = i.find(' ')
#         key = i[0:p]
#         value = i[p + 1:]
#         # print('${0}$'.format(value))
#         dic[key] = [value]
#     return dic
#
# class random_uniform_num():
#
#     """
#     均匀随机，确保每轮每个只出现一次
#     """
#     def __init__(self, total):
#         self.total = total
#         self.range = [i for i in range(total)]
#         np.random.shuffle(self.range)
#         self.index = 0
#
#     def get(self, batchsize):
#         r_n = []
#         if (self.index + batchsize > self.total):
#             r_n_1 = self.range[self.index:self.total]
#             np.random.shuffle(self.range)
#             self.index = (self.index + batchsize) - self.total
#             r_n_2 = self.range[0:self.index]
#             r_n.extend(r_n_1)
#             r_n.extend(r_n_2)
#         else:
#             r_n = self.range[self.index: self.index + batchsize]
#             self.index = self.index + batchsize
#
#         return r_n

#将字符转化为标签
def getlabels(text):
    text = text[0]
    labels = []
    for char in text:
        labels.append(chardic[char])
    return labels

def charfind(x):
    if x in char_set:
        return char_set.find(x)
    else:
        return 0

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.crnn2(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])

    return basemodel, model

if __name__ == '__main__':

    K.set_session(get_session(1))
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    # 预训练权重 第3轮 更换优化器 adam ->RMSprop
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0911_name_only_03-loss-23.493181.h5'
    # 第40轮 更换 Rmsprop-> adam
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0911_name_only_40-loss-1.463713.h5'

    # 更换字体，减小字号，名字之间增加空格，加大模概率(重新训练）
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0911_name_only_71-loss-0.294086-acc-0.8836.h5'
    # 调整文字大小（随机），加大模糊率
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0912_name_only_40-loss-0.043436-acc-0.9947.h5'

    # 增加DF印章图片（随机比例20%），重新训练
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0912_name_only_85-loss-0.336846-acc-0.8518.h5' # 0.03528200
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0913_name_only_36-loss-0.431686-acc-0.9234.h5'  # 0.0916490
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0913_name_only_66-loss-0.270262-acc-0.9508.h5'  # 0.0940050
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0913_name_only_69-loss-0.273818-acc-0.9486.h5' # 0.0952580
    # 增加 unet 处理后的df 图片
    # modelPath = 'E:/projects/python/DF_idcard_all/train_model/name_weights/0913_name_only_130-loss-0.220576-acc-0.9603.h5' #0.0961530

    # 更换新工程重新训练（在0913_name_only_292-loss-0.187381-acc-0.9662.h5基础上继续训练）
    # modelPath = WEIGHTS_PATH + '/0913_name_only_292-loss-0.187381-acc-0.9662.h5'
    #160轮继续训练
    # modelPath = WEIGHTS_PATH + '/1009_name_only_160-loss-0.074459-acc-0.9818.h5'


    # 几乎扩大训练集600w，增加然后的名片图片，提高分数
    # modelPath = WEIGHTS_PATH + '/1009_name_only_400-loss-0.054976-acc-0.9882.h5'
    # modelPath = WEIGHTS_PATH + '/1013_name_only_67-loss-0.089308-acc-0.9746.h5' #中断后继续训练
    modelPath = WEIGHTS_PATH + '/1013_name_only_306-loss-0.048609-acc-0.9880.h5'  # 中断后继续训练
    start_epochs = 306

    model.load_weights(modelPath)

    training_generator = FileImageDataGenerator(data_size=200000,
                                                corpus_file=CHAR_FILE_PATH,
                                                max_label_length=maxlabellength,
                                                train_dir=DATA_PATH + '/out_img',
                                                train_label=DATA_PATH + '/train_maker_label.txt',
                                                df_train_dir=DATA_PATH + '/train_img',
                                                df_train_label=DATA_PATH + '/name_lable_by_df_train.txt',
                                                batch_size=batch_size,
                                                image_size=[img_h, img_w],
                                                df_data_ratio=0.05  # data_szie记录的10%(1w）
                                                )

    test_generator = FileImageDataGenerator(data_size=10000,
                                            corpus_file=CHAR_FILE_PATH,
                                            max_label_length=maxlabellength,
                                            train_dir=DATA_PATH + '/out_img',
                                            train_label=DATA_PATH + '/train_maker_label.txt',
                                            df_train_dir=DATA_PATH + '/train_img',
                                            df_train_label=DATA_PATH + '/name_lable_by_df_train.txt',
                                            batch_size=batch_size,
                                            image_size=[img_h, img_w],
                                            df_data_ratio=0.5)

    # -------AutoImageDataGenerator 废除-------
    # training_generator = AutoImageDataGenerator(data_size = 200000,
    #                                             corpus_file = CHAR_FILE_PATH,
    #                                             df_train_dir = DF_DATA_PATH + '/train_img',
    #                                             df_train_label = DF_DATA_PATH +'/name_lable_by_df_train.txt',
    #                                             max_label_length = 4,
    #                                             batch_size = batch_size,
    #                                             image_size = [img_h, img_w],
    #                                             df_data_ratio = 0.05) # 1W 记录
    #
    # test_generator = AutoImageDataGenerator(data_size = 10000,
    #                                             corpus_file=CHAR_FILE_PATH,
    #                                             df_train_dir = DF_DATA_PATH + '/train_img',
    #                                             df_train_label = DF_DATA_PATH +'/name_lable_by_df_train.txt',
    #                                             max_label_length = 4,
    #                                             batch_size = batch_size,
    #                                             image_size =[img_h, img_w],
    #                                             df_data_ratio=0.5)

    checkpoint = ModelCheckpoint(filepath=WEIGHTS_PATH + '/'+ DATE_VER +'_only_{epoch:02d}-loss-{val_loss:.6f}-acc-{val_acc:.4f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)

    lr_schedule = lambda epoch: 0.0001 * 0.99 ** epoch

    learning_rate = np.array([lr_schedule(i) for i in range(400)])
    print(learning_rate)
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=400, verbose=1)
    tensorboard = TensorBoard(log_dir=WEIGHTS_PATH + '/logs', write_graph=True)


    print('-----------Start training-----------')
    model.fit_generator(training_generator,
                        # steps_per_epoch= 50000 // batch_size,
                        epochs=400,
                        initial_epoch=start_epochs,
                        validation_data=test_generator,
                        # validation_steps=10000 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard],
                        workers=2
                        )
