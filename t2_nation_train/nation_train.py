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
import pandas as pd


# 设置基本参数
# sizew = 300
# sizeh = 90
sizew = 120*2
sizeh = 40*2

# batch_size = 16
batch_size = 256

n_class = 56

# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/nation_weights'

# 源路径：下层文件夹是民族子文件夹
# src_path = "E:/projects/python/DF_idcard_all/temp/nationality"
src_path = "D:/projects/data/ccf2019_ocr/nation_l2"

# 训练权重日期版本（不要重复）
DATE_VER = '1018_v5'+ '_nation'

# 民族名称list，对应文件夹名字正好是list的下标
nation_list = ['土家', '塔吉克', '汉', '佤', '撒拉', '白', '高山', '傣', '纳西', '鄂温克'
               , '畲', '回', '藏', '傈僳', '达斡尔', '锡伯', '塔塔尔', '保安', '独龙', '珞巴'
               , '布依', '景颇', '赫哲', '基诺', '侗', '乌孜别克', '布朗', '门巴', '崩龙', '土'
               , '朝鲜', '蒙古', '东乡', '哈萨克', '黎', '水', '仫佬', '京', '怒', '阿昌'
               , '哈尼', '拉祜', '羌', '柯尔克孜', '满', '裕固','毛难', '瑶', '维吾尔', '壮'
               , '鄂伦春', '俄罗斯', '仡佬', '彝', '苗', '普米']


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

# def process_line(line):
#     tmp = line.strip().split(' ')
#     x = tmp[0]
#     y = tmp[1]
#     return x, y
#
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

imglist = []   # 全部图片的list
img_label_dict = {}    # 全部 图片名-label的dict

b = list(range(56))     # 56类
labels = to_categorical(b)   # 标签序列化

# ======================= DF 训练数据=========================
g = os.walk(src_path + '/train_img')
# 遍历路径下文件夹: path, dir_list, file_list分别为源路径，路径下的文件夹列表，路径下的文件列表
for path,dir_list,file_list in g:
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        child_imglist = walkpath(dir_path, 'jpg')    # 文件夹下所有jpg文件
        imglist += child_imglist     # 存入全部图片list中
        for img_name in child_imglist:
            img_label_dict[img_name] = labels[int(dir)]

# ======================= ImageMaker 训练数据=========================
maker_train_img = pd.read_csv(src_path + '/train_maker_label.txt',sep=" ", header=None, names=['file', 'nation_idx'])
for idx, row in maker_train_img.iterrows():
    file_path = '{0}/{1}.jpg'.format(src_path + '/out_img',row['file'])
    imglist.append(file_path)
    img_label_dict[file_path] = labels[int(row['nation_idx'])]

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

# # ======================= ImageMaker 训练数据=========================
# maker_train_img = pd.read_csv(src_path + '/train_maker_label.txt',sep=" ", header=None, names=['file', 'nation_idx'])
# maker_imglist = []   # 全部图片的list
# maker_img_label_dict = {}    # 全部 图片名-label的dict
#
# for idx, row in maker_train_img.iterrows():
#     maker_imglist.append(row['file'])
#     maker_img_label_dict[row['file']] = labels[int(row['nation_idx'])]
#
# def generate_arrays_from_file_by_maker():
#     # ------数据生成器------
#     while 1:
#         cnt = 0
#         path_temp = random.sample(maker_imglist, batch_size)
#         X = []
#         Y = []
#         for path in path_temp:
#             file_path = '{0}/{1}.jpg'.format(src_path + '/out_img',path)
#             # print(file_path)
#             x = cv.imread(file_path)
#             x = cv.resize(x, (sizew, sizeh))  # (width, height)
#
#             X.append(x)
#             y = maker_img_label_dict[path]
#             Y.append(y)
#             cnt += 1
#             if cnt == batch_size:
#                 cnt = 0
#                 X = np.array(X)
#                 X = X / 255.0
#                 Y = np.array(Y)
#                 yield (X, Y)
#                 X = []
#                 Y = []

model = creatmodel()
# model_path = WEIGHTS_PATH + '/0905_mix_ext1w_nation_weights-net-0114-0.000222.h5' # 添加新数据重新训练
# model_path = WEIGHTS_PATH + '/1018_nation_only_55-loss-0.006415-acc-0.9979.h5'  # 继续添加120w数据训练
# model_path = WEIGHTS_PATH + '/1018_v2_nation_only_55-loss-0.002515-acc-0.9994.h5'  # 继续添加6w(易错 回,哈萨克,拉祜)数据训练
# model_path = WEIGHTS_PATH + '/1018_v3_nation_only_37-loss-0.001533-acc-0.9996.h5'  # 继续添加12w(易错 塔吉克,回,侗,哈萨克,拉祜)数据训练
model_path = WEIGHTS_PATH + '/1018_v4_nation_only_42-loss-0.000463-acc-0.9999.h5'  # 继续添加12w(易错 朝鲜,回,景颇)数据训练

start_epochs = 0
model.load_weights(model_path)

# lr_schedule = lambda epoch: 0.001 * 0.95 ** epoch
lr_schedule = lambda epoch: 0.0001 * 0.95 ** epoch # 新训练的学习率

learning_rate = np.array([lr_schedule(i) for i in range(100)])
changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
tensorboard = TensorBoard(log_dir= WEIGHTS_PATH + '/logs', write_graph=True)

print('-----------Start training-----------')

# train_loader = generate_arrays_from_file_by_maker() # 生成的数据
train_loader = generate_arrays_from_file() # 生成的数据
test_loader = generate_arrays_from_file() # 官方提供数据作为验证集


checkpoint = ModelCheckpoint(filepath= WEIGHTS_PATH + '/' + DATE_VER + '_only_{epoch:02d}-loss-{val_loss:.6f}-acc-{val_acc:.4f}.h5', monitor='val_loss',
                             save_best_only=False, save_weights_only=True)

model.fit_generator(train_loader,
                    steps_per_epoch = 50000 // batch_size,
                    epochs = 100 ,
                    initial_epoch = start_epochs,
                    validation_data = test_loader,
                    validation_steps = 10000 // batch_size,
                    callbacks=[checkpoint, earlystop, changelr, tensorboard])


