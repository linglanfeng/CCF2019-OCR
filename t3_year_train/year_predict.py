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

# 设置基本参数
sizew = 70 * 2
sizeh = 40 * 2
batch_size = 32
# batch_size = 16
n_class = 0

# 源训练集路径
# src_path = "E:/projects/python/DF_idcard_all/temp/year"

# # 源路径：下面是子文件夹
# src_path = "E:/projects/python/DF_idcard_all/temp/test"
# output_path = 'result/year_result_mix_ext1w.csv'

src_path = "D:/projects/data/ccf2019_ocr/year_l2/test_img"
output_path = '../single_result/1019_year_only_31.csv'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/year_weights'

# model_path = WEIGHTS_PATH + '/year_mix_ext1w_weights-net-0067-0.00093553.h5'
model_path = WEIGHTS_PATH + '/1019_year_only_31-loss-0.009296-acc-0.9974.h5'
# model_path = WEIGHTS_PATH + '/1019_v2_year_only_42-loss-0.001413-acc-0.9997.h5' # 错误4个不如 训练前。

# g = os.walk(src_path)  # 遍历路径
# img_dir_list = []   #有图片的文件夹路径list，也作为标签list
#
# # 判断文件夹中是否有文件，计算出类别数
# for path,dir_list,file_list in g:
#     for dir in dir_list:
#         if not os.listdir(src_path + '/' + dir):
#             continue
#         n_class += 1
#         img_dir_list.append(dir)

# 类别
img_dir_list = []   #有图片的文件夹路径list，也作为标签list
for i in range(1958,2009): # 训练集范围
    n_class += 1
    img_dir_list.append(str(i))


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

def creatmodel():
    # Transfer learning with Inception V3
    # base_model = applications.MobileNetV2(include_top=False, input_shape=(sizew, sizeh, 3), weights=None)
    base_model = applications.InceptionV3(include_top=False, weights=None, input_shape=(sizeh, sizew, 3))    # (高，宽，通道数)

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

# 测试样本图片路径
all_imgs = walkpath(src_path, 'jpg')    # 文件夹下所有文件
test_imgs = [x for x in all_imgs if x.split('/')[-1][-5]=='3']

# 新建并载入模型
model = creatmodel()
model.load_weights(model_path)


predict_Y = []
img_names = []
# score = []
for path in test_imgs:
    x = cv.imread(path)
    if x is not None:
        x = cv.resize(x, (sizew, sizeh))  # (width, height)
        X = [x]
        X = np.array(X)
        X = X / 255.0
        y = model.predict(X)
        y_label = np.argmax(y)   # 数字label
        y_str = img_dir_list[y_label]    # 字符label
        print(path+' : '+str(y_str),np.max(y))
        predict_Y.append(y_str)
        # score.append(np.max(y))
        # img_names.append(path.split('/')[-1][:-6])
        img_names.append(path.replace('\\', '/').split('/')[-1][:-6])

# c={"name": img_names, "predict" : predict_Y, "score": score}
c={"name": img_names, "predict" : predict_Y}

import pandas as pd
df = pd.DataFrame(c)

df.to_csv(output_path, encoding='utf_8_sig', header=None, index=False)

