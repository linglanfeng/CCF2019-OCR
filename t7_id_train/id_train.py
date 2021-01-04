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

CHAR_FILE_PATH = '../ext_data/corpus/char(id).txt'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/id_weights'

DATA_PATH = 'D:/projects/data/ccf2019_ocr/id_l2'

# 训练权重日期版本（不要重复）
DATE_VER = '1021_v3'+ '_id'

image_resize_times = 1
img_h = int(40 * image_resize_times)
img_w = int(300 * image_resize_times)

batch_size = 512
maxlabellength = 18

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

    # 几乎扩大训练集100w，增加然后的名片图片，提高分数
    # modelPath = WEIGHTS_PATH + '/crnn2-50-0.096743-acc_1.000000.h5'
    # modelPath = WEIGHTS_PATH + '/1015_id_only_100-loss-0.001126-acc-0.9999.h5' # 增加数据再次训练 100次 降低至 50次
    # modelPath = WEIGHTS_PATH + '/1015_id_only_100-loss-0.001126-acc-0.9999.h5'  # 增加数据再次训练 100次 降低至 50次
    # modelPath = WEIGHTS_PATH + '/1015_id_only_50-loss-0.003156-acc-0.9991.h5'  # 增加印章遮挡概率（120w），重新训练 放弃V3版本，增加早停 效果不错，33
    modelPath = WEIGHTS_PATH + '/1021_v2_id_only_100-loss-0.000738-acc-0.9996.h5'  # 增加印章遮挡概率（60w+60w），增加数据继续训练 v2版本效果不好，继续训练V3
    start_epochs = 0

    model.load_weights(modelPath)

    training_generator = FileImageDataGenerator(data_size=100000,
                                                corpus_file=CHAR_FILE_PATH,
                                                max_label_length=maxlabellength,
                                                train_dir=DATA_PATH + '/out_img',
                                                train_label=DATA_PATH + '/train_maker_label.txt',
                                                df_train_dir=DATA_PATH + '/train_img',
                                                df_train_label=DATA_PATH + '/id_lable_by_df_train.txt',
                                                batch_size=batch_size,
                                                image_size=[img_h, img_w],
                                                df_data_ratio=0.1  # data_szie记录的10%(1w）
                                                )

    test_generator = FileImageDataGenerator(data_size=20000, # 1022 1w->2w
                                            corpus_file=CHAR_FILE_PATH,
                                            max_label_length=maxlabellength,
                                            train_dir=DATA_PATH + '/out_img',
                                            train_label=DATA_PATH + '/train_maker_label.txt',
                                            df_train_dir=DATA_PATH + '/train_img',
                                            df_train_label=DATA_PATH + '/id_lable_by_df_train.txt',
                                            batch_size=batch_size,
                                            image_size=[img_h, img_w],
                                            df_data_ratio=0.5)
                                            # df_data_ratio=0.9) # 提高val比例至90% 分数没有变化

    checkpoint = ModelCheckpoint(filepath=WEIGHTS_PATH + '/'+ DATE_VER +'_only_{epoch:02d}-loss-{val_loss:.6f}-acc-{val_acc:.4f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)

    # lr_schedule = lambda epoch: 0.0001 * 0.99 ** epoch
    lr_schedule = lambda epoch: 0.0001 * 0.95 ** epoch

    learning_rate = np.array([lr_schedule(i) for i in range(100)])
    print(learning_rate)
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    # earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    tensorboard = TensorBoard(log_dir=WEIGHTS_PATH + '/logs', write_graph=True)


    print('-----------Start training-----------')
    model.fit_generator(training_generator,
                        # steps_per_epoch= 50000 // batch_size,
                        epochs=100,
                        initial_epoch=start_epochs,
                        validation_data=test_generator,
                        # validation_steps=10000 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard],
                        workers=2
                        )
