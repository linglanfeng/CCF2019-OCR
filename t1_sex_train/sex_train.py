import os
from keras import layers,models
import numpy as np
import cv2 as cv
from keras.utils import np_utils, to_categorical
from keras import applications
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

import math
import random
from keras.layers.pooling import GlobalAveragePooling2D

# 设置基本参数   文件大小为w*h 40*30
sizew = 60*2
sizeh = 40*2
batch_size = 32

n_class = 2 # [男、女]

def creatmodel():
    # Transfer learning with Inception V3
    # base_model = applications.MobileNetV2(include_top=False, input_shape=(sizew, sizeh, 3), weights=None)
    base_model = applications.MobileNetV2(include_top=False, weights=None, input_shape=(sizeh, sizew, 3))    # (高，宽，通道数)

    x = base_model.output
    # 每个特征图平均成一个特征 也是展平的效果
    x = GlobalAveragePooling2D()(x)

    # 全连接网络
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(n_class, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def process_line(line):
    tmp = line.strip().split(' ')
    x = tmp[0]
    y = tmp[1]
    return x, y

# def rotateImage(src, degree):
#     # 旋转中心为图像中心
#     h, w = src.shape[:2]
#     # 计算二维旋转的仿射变换矩阵
#     RotateMatrix = cv.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
#     #print(RotateMatrix)
#     # 仿射变换，背景色填充为白色
#     #取最大的长和宽
#     L = max(w,h)
#     rotate = cv.warpAffine(src, RotateMatrix, (L,L), borderValue=(255, 255, 255))
#     return rotate

def walkpath(rootpath,fileclass):
    #------找出路径下某一类型的所有文件，输出为list------
    pathlist = []
    list = os.listdir(rootpath)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootpath, list[i])
        if os.path.isfile(path):
            if fileclass in path:
                pathlist.append(path)
    return pathlist


# 源路径：下面是民族子文件夹
src_path = "E:/projects/python/DF_idcard_all/temp/sex"

# 民族名称list，对应文件夹名字正好是list的下标

imglist = []   # 全部图片的list
img_label_dict = {}    # 全部 图片名-label的dict

b = list(range(n_class))     # 2类
labels = to_categorical(b)   # 标签序列化

g = os.walk(src_path)
# 遍历路径下文件夹: path, dir_list, file_list分别为源路径，（路径下）文件夹列表，（路径下）文件列表
for path,dir_list,file_list in g:
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        child_imglist = walkpath(dir_path, 'jpg')    # 文件夹下所有jpg文件
        imglist += child_imglist     # 存入全部图片list中
        for img_name in child_imglist:
            img_label_dict[img_name] = labels[int(dir)]


def generate_arrays_from_file():
    # ------数据生成器------
    while 1:
        cnt = 0
        path_temp = random.sample(imglist, batch_size)
        X = []
        Y = []
        for path in path_temp:

            x = cv.imread(path)
            x = cv.resize(x, (sizew, sizeh))  # (width, height)

            X.append(x)
            y = img_label_dict[path]
            Y.append(y)
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                X = np.array(X)
                X = X / 255.0
                Y = np.array(Y)
                yield (X, Y)
                X = []
                Y = []

# model_path = 'modelangle/weights-net-09-0.00.h5'
model = creatmodel()
# model.load_weights(model_path)

lr_schedule = lambda epoch: 0.001 * 0.95 ** epoch
learning_rate = np.array([lr_schedule(i) for i in range(100)])
changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
tensorboard = TensorBoard(log_dir='train_model/sex_weights/logs', write_graph=True)

print('-----------Start training-----------')

train_loader = generate_arrays_from_file()
test_loader = generate_arrays_from_file()

checkpoint = ModelCheckpoint(filepath='train_model/sex_weights/sex_mix_weights-net-{epoch:04d}-{val_loss:.8f}.h5', monitor='val_loss',
                             save_best_only=False, save_weights_only=True)

model.fit_generator(train_loader,
                    steps_per_epoch=10000 // batch_size,
                    epochs=100,
                    initial_epoch=0,
                    validation_data=test_loader,
                    validation_steps=1000,
                    callbacks=[checkpoint, earlystop, changelr, tensorboard])


