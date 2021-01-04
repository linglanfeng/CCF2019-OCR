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
import id_train.densenet as densenet

# 300*40像素
img_h = 40
img_w = 300
batch_size = 32

maxlabellength = 18

chardic = {}
numdic ={}
charfilepath = 'id_char.txt'

charfile = open(charfilepath,encoding='utf8')
lines = charfile.readlines()
char_set = open(charfilepath, 'r', encoding='utf-8').readlines()
char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
nclass = len(char_set)
str = '0542'
print([char_set.find(x) for x in str])
for line in lines:
    line = line.replace('\n','')
    #key是字符
    chardic[line] = lines.index(line+'\n')

for line in lines:
    line = line.replace('\n','')
    numdic[lines.index(line+'\n')] = line

def get_session(gpu_fraction=0.7):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.find(' ')
        key = i[0:p]
        # i[p + 1:] -> i[p + 2:]
        value = i[p + 2:]
        dic[key] = [value]
    return dic
class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

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

def gen(data_file, image_path, batchsize=128, maxlabellength=4, imagesize=(32, 128)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            if (len(image_label[j][0]) <= 0):
                continue
            img1 = Image.open(os.path.join(image_path, j)+'.jpg').convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)

            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str[0])
            input_length[i] = imagesize[1] // 4 + 1
            #将str里面的字符转化为数字标签
            # print("${0}$".format(str[0]), len(str[0]))
            labels[i,:len(str[0])] = [charfind(x) for x in str[0]]

            if (len(str[0]) <= 0):
                print("len < 0", j)

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


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
    K.set_session(get_session())
    reload(densenet)
    basemodel, model = get_model(img_h, nclass)

    # modelPath = './model/weights-densenet-10-0.43.h5'
    # if os.path.exists(modelPath):
    #     print("Loading model weights...")
    #     model.load_weights(modelPath)
    #     print('done!')

    model.load_weights('E:/projects/python/DF_idcard_all/train_model/id_weights/crnn2-50-0.096743-acc_1.000000.h5') # 扩展1w数据后继续训练
    start_epoch = 0

    train_loader = gen('E:/projects/python/DF_idcard_all/temp/tmp_labels.txt', 'E:/projects/python/DF_idcard_all/temp/id', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))
    test_loader = gen('E:/projects/python/DF_idcard_all/temp/tmp_labels.txt', 'E:/projects/python/DF_idcard_all/temp/id', batchsize=batch_size, maxlabellength=maxlabellength, imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='E:/projects/python/DF_idcard_all/train_model/id_weights/id_mix_ext1w_weights-{epoch:02d}-{val_loss:.6f}-acc_{val_acc:.6f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)

    lr_schedule = lambda epoch: 0.001 * 0.9 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(100)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir='E:/projects/python/DF_idcard_all/train_model/id_weights/logs', write_graph=True)
    # model.save_weights('model/sfz-weights.h5')

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=18000 // batch_size,
                        epochs=100,
                        initial_epoch=start_epoch,
                        validation_data=test_loader,
                        validation_steps=2000 // batch_size, # 扩展数据后 1000->2000
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])
