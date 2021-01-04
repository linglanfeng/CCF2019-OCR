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
sizew = 120*2
sizeh = 40*2

# batch_size = 16
batch_size = 32
n_class = 56

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


# 源路径
# src_path = "E:/projects/python/DF_idcard_all/temp/test_unet"
# output_path = 'result/nation_result_mix_ext1w_unet_v2.csv'

src_path = "D:/projects/data/ccf2019_ocr/nation_l2/test_img"
output_path = '../single_result/1018_v5_nation_only_46.csv'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/nation_weights'

# model_path = './train_model/nation_weights-net-0056-0.000011.h5'
# model_path = './train_model/0904_mix_nation_weights-net-0053-0.000007.h5'
# model_path = './train_model/0905_mix_ext1w_nation_weights-net-0097-0.000255.h5'

# model_path = WEIGHTS_PATH + '/0905_mix_ext1w_nation_weights-net-0114-0.000222.h5'
model_path = WEIGHTS_PATH + '/1018_v5_nation_only_46-loss-0.000274-acc-0.9999.h5'

# 民族名称list，对应文件夹名字正好是list的下标
nation_list = ['土家', '塔吉克', '汉', '佤', '撒拉', '白', '高山', '傣', '纳西', '鄂温克'
               , '畲', '回', '藏', '傈僳', '达斡尔', '锡伯', '塔塔尔', '保安', '独龙', '珞巴'
               , '布依', '景颇', '赫哲', '基诺', '侗', '乌孜别克', '布朗', '门巴', '崩龙', '土'
               , '朝鲜', '蒙古', '东乡', '哈萨克', '黎', '水', '仫佬', '京', '怒', '阿昌'
               , '哈尼', '拉祜', '羌', '柯尔克孜', '满', '裕固','毛难', '瑶', '维吾尔', '壮'
               , '鄂伦春', '俄罗斯', '仡佬', '彝', '苗', '普米']


# 测试样本图片路径
all_imgs = walkpath(src_path, 'jpg')    # 文件夹下所有文件
test_imgs = [x for x in all_imgs if x.split('/')[-1][-5]=='2']

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
        y_str = nation_list[y_label]    # 民族字符label
        print(path+' : '+str(y_str), np.max(y))
        predict_Y.append(y_str)
        # score.append(np.max(y))
        # img_names.append(path.split('/')[-1][:-6])
        img_names.append(path.replace('\\', '/').split('/')[-1][:-6])

# c={"name": img_names, "predict" : predict_Y, "score": score}
c={"name": img_names, "predict" : predict_Y}

import pandas as pd
df = pd.DataFrame(c)

df.to_csv(output_path, encoding='utf_8_sig', header=None, index=False)

print()

