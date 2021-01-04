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

CHAR_FILE_PATH = '../ext_data/corpus/char(valid).txt'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/valid_weights'

DATA_PATH = 'D:/projects/data/ccf2019_ocr/valid_l2'

# 训练权重日期版本（不要重复）
DATE_VER = '1020_v4'+ '_valid'

image_resize_times = 1
img_h = int(35 * image_resize_times)
img_w = int(240 * image_resize_times)

batch_size = 1024
maxlabellength = 21

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

    # 从头训练，图片缩小范围（去掉无用下边，更换decode函数，不放大图片增加batchsize）

    # 中断继续训练
    # modelPath = WEIGHTS_PATH + '/1012_valid_only_05-loss-0.267624-acc-0.9337.h5'
    # modelPath = WEIGHTS_PATH + '/1012_valid_only_35-loss-0.016823-acc-0.9938.h5'

    # 增加数据继续训练（100轮开始）
    # modelPath = WEIGHTS_PATH + '/1012_valid_only_100-loss-0.000817-acc-0.9999.h5'

    # 重新生成数据（120w）增强印章出现频率 重新生成数据训练，设置早停
    #modelPath = WEIGHTS_PATH + '/1012_valid_only_200-loss-0.000293-acc-0.9999.h5'
    # modelPath = WEIGHTS_PATH + '/1020_valid_only_108-loss-0.002840-acc-0.9988.h5' # 早停后对比结果，效果一般，继续增加60w数据训练
    # modelPath = WEIGHTS_PATH + '/1020_v2_valid_only_85-loss-0.000553-acc-0.9997.h5' # 停机再训练 早停临时20->10  ，79轮最好，100轮停止，更好。
    # modelPath = WEIGHTS_PATH + '/1020_v2_valid_only_100-loss-0.000226-acc-1.0000.h5'  # 再次加入60w数据，继续训练，早停设置10。 效果还行，还有几幅有明显错误 1020_v3_valid_only_39-loss-0.000628-acc-0.9998.h5
    modelPath = WEIGHTS_PATH + '/1020_v3_valid_only_39-loss-0.000628-acc-0.9998.h5'  # 再次加入60w数据，继续训练，早停设置10。训练第4版
    start_epochs = 0

    model.load_weights(modelPath)

    # training_generator = FileImageDataGenerator(data_size=200000,
    training_generator = FileImageDataGenerator(data_size=100000, # 1020日 20w -> 10w
                                                corpus_file=CHAR_FILE_PATH,
                                                max_label_length=maxlabellength,
                                                train_dir=DATA_PATH + '/out_img',
                                                train_label=DATA_PATH + '/train_maker_label.txt',
                                                df_train_dir=DATA_PATH + '/train_img',
                                                df_train_label=DATA_PATH + '/valid_lable_by_df_train.txt',
                                                batch_size=batch_size,
                                                image_size=[img_h, img_w],
                                                df_data_ratio=0.05  # data_szie记录的10%(1w）
                                                # df_data_ratio = 0,
                                                # df_train_dir = DATA_PATH + '/out_img',
                                                # df_train_label = DATA_PATH + '/train_maker_label.txt',
                                                )

    test_generator = FileImageDataGenerator(data_size=10000,
                                            corpus_file=CHAR_FILE_PATH,
                                            max_label_length=maxlabellength,
                                            train_dir=DATA_PATH + '/out_img',
                                            train_label=DATA_PATH + '/train_maker_label.txt',
                                            df_train_dir=DATA_PATH + '/train_img',
                                            df_train_label=DATA_PATH + '/valid_lable_by_df_train.txt',
                                            batch_size=batch_size,
                                            image_size=[img_h, img_w],
                                            df_data_ratio=0.5
                                            # df_data_ratio=0,
                                            # df_train_dir=DATA_PATH + '/out_img',
                                            # df_train_label=DATA_PATH + '/train_maker_label.txt',
                                            )

    checkpoint = ModelCheckpoint(filepath=WEIGHTS_PATH + '/'+ DATE_VER +'_only_{epoch:02d}-loss-{val_loss:.6f}-acc-{val_acc:.4f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)

    lr_schedule = lambda epoch: 0.0001 * 0.99 ** epoch

    learning_rate = np.array([lr_schedule(i) for i in range(100)])
    print(learning_rate)
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=WEIGHTS_PATH + '/logs', write_graph=True)


    print('-----------Start training-----------')
    model.fit_generator(training_generator,
                        # steps_per_epoch= 50000 // batch_size,
                        epochs = 100,
                        initial_epoch=start_epochs,
                        validation_data=test_generator,
                        # validation_steps=10000 // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard],
                        workers=2
                        )
