from keras import backend as K
import cv2
import os
import re
import csv
import numpy as np
import pandas as pd
from keras.utils import multi_gpu_model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Dense
from keras.models import save_model, load_model, Model
from keras import applications
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Reshape, Masking, Lambda, Permute
from PIL import Image, ImageDraw, ImageFont
from yolo3.model_vgg16 import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
# from ctpn.model import bekenctpn as ctpn
from prepare.image_rectification import jiaozheng, rotate_bound
from crnn import densenet


class YOLO(object):
    """
    YOLO模型
    """
    def __init__(self):
        self.model_path = 'best_model/trained_weights_final.h5' # model path or trained weights path
        self.anchors_path = 'best_model/yolo_anchors.txt'
        self.classes_path = 'best_model/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.gpu_num = 1
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        # default arg
        # self.yolo_model->'model_data/yolo.h5'
        # self.anchors->'model_data/yolo_anchors.txt'-> 9 scales for anchors
        return boxes, scores, classes

    def detect_image(self, image):
        # start = timer()
        rects = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tf.Session.run(fetches, feed_dict=None)
        # Runs the operations and evaluates the tensors in fetches.
        #
        # Args:
        # fetches: A single graph element, or a list of graph elements(described above).
        #
        # feed_dict: A dictionary that maps graph elements to values(described above).
        #
        # Returns:Either a single value if fetches is a single graph element, or a
        # list of values if fetches is a list(described above).
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #             size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 500

        classes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 0.5).astype('float32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('float32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('float32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('float32'))
            # print(label, (x1, y1), (x2, y2))
            bbox = dict([("score",str(score)),("x1",str(x1)),("y1", str(y1)),("x2", str(x2)),("y2", str(y2))])
            rects.append(bbox)
            classes.append(predicted_class)

        return rects, classes

    def close_session(self):
        self.sess.close()


class Unet(object):
    """
    Unet模型
    """
    def __init__(self):
        self.model_path = 'best_model/unet_weights.h5' # model path or trained weights path
        self.input_size_1 = 1024
        self.input_size_2 = 1024
        self.model = self.create_model()
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    def Conv2d_BN(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def Conv2dT_BN(self, x, filters, kernel_size, strides=(2, 2), padding='same'):
        x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def create_model(self):
        inpt = Input(shape=(self.input_size_1, self.input_size_2, 3))

        conv1 = self.Conv2d_BN(inpt, 8, (3, 3))
        conv1 = self.Conv2d_BN(conv1, 8, (3, 3))
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

        conv2 = self.Conv2d_BN(pool1, 16, (3, 3))
        conv2 = self.Conv2d_BN(conv2, 16, (3, 3))
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

        conv3 = self.Conv2d_BN(pool2, 32, (3, 3))
        conv3 = self.Conv2d_BN(conv3, 32, (3, 3))
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

        conv4 = self.Conv2d_BN(pool3, 64, (3, 3))
        conv4 = self.Conv2d_BN(conv4, 64, (3, 3))
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

        conv5 = self.Conv2d_BN(pool4, 128, (3, 3))
        # conv5 = Dropout(0.1)(conv5)
        conv5 = self.Conv2d_BN(conv5, 128, (3, 3))
        # conv5 = Dropout(0.1)(conv5)

        convt1 = self.Conv2dT_BN(conv5, 64, (3, 3))
        concat1 = concatenate([conv4, convt1], axis=3)
        # concat1 = Dropout(0.1)(concat1)
        conv6 = self.Conv2d_BN(concat1, 64, (3, 3))
        conv6 = self.Conv2d_BN(conv6, 64, (3, 3))

        convt2 = self.Conv2dT_BN(conv6, 32, (3, 3))
        concat2 = concatenate([conv3, convt2], axis=3)
        # concat2 = Dropout(0.1)(concat2)
        conv7 = self.Conv2d_BN(concat2, 32, (3, 3))
        conv7 = self.Conv2d_BN(conv7, 32, (3, 3))

        convt3 = self.Conv2dT_BN(conv7, 16, (3, 3))
        concat3 = concatenate([conv2, convt3], axis=3)
        # concat3 = Dropout(0.1)(concat3)
        conv8 = self.Conv2d_BN(concat3, 16, (3, 3))
        conv8 = self.Conv2d_BN(conv8, 16, (3, 3))

        convt4 = self.Conv2dT_BN(conv8, 8, (3, 3))
        concat4 = concatenate([conv1, convt4], axis=3)
        # concat4 = Dropout(0.1)(concat4)
        conv9 = self.Conv2d_BN(concat4, 8, (3, 3))
        conv9 = self.Conv2d_BN(conv9, 8, (3, 3))
        # conv9 = Dropout(0.1)(conv9)
        outpt = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)

        model = Model(inpt, outpt)
        model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae'])
        # model.summary()
        return model

    def predict(self, path):
        imgorg = cv2.imread(path)
        imgorg = cv2.resize(imgorg, (self.input_size_1, self.input_size_2))
        imgorg = np.array(imgorg, 'f') / 255.0
        x = np.array([imgorg])
        y = self.model.predict(x)[0] * 255.0
        return y

# class CPTN(object):
#     """
#     cptn模型
#     """
#     def __init__(self):
#         self.VGG_PATH = 'model/vgg16_weights.h5'
#         self.CTPN_MODEL_PATH = 'model/ctpn_weights.hdf5'
#         self.fold_path = 'output_imgs'
#         self.model = ctpn.Ctpn()
#         self.model.bulid_model(self.VGG_PATH)
#         self.model.load_model(self.CTPN_MODEL_PATH)
#
#     def predict(self, img_org, img_name):
#         """
#         cptn内部定位
#         :param imgorg:
#         :param ctpn_model:
#         :param savepath:
#         :return:
#         """
#         img, text, scale, angle = self.model.predict(img_org)
#
#         img_copy = img_org.copy()
#
#         rec_list = []   # 全部定位框的数组
#         for box in text:
#             dict = {}
#             w = int((box[2] - box[0]) / scale)
#             h = int((box[5] - box[1]) / scale)
#             POINT_y = int(box[1] / scale)
#             POINT_x = int(box[0] / scale)
#
#             cv2.rectangle(img_copy, (POINT_x, POINT_y), (POINT_x + w, POINT_y + h), (0, 255, 0), 2)
#             rec_list.append([POINT_x, POINT_y, POINT_x + w, POINT_y + h])
#
#         save_path = self.fold_path + '/' + img_name
#         cv2.imwrite(save_path, img_copy)
#         return rec_list


class Classify(object):
    """
    分类模型
    """
    def __init__(self, model_path):
        self.sizew = 480
        self.sizeh = 350
        self.n_class = 2
        self.model_path = model_path
        self.model = self.create_model()
        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

    def create_model(self):
        # Transfer learning with Inception V3
        # base_model = applications.MobileNetV2(include_top=False, input_shape=(self.sizew, self.sizeh, 3), weights=None)
        base_model = applications.MobileNetV2(include_top=False, weights=None,
                                              input_shape=(self.sizeh, self.sizew, 3))  # (高，宽，通道数)

        x = base_model.output
        # 每个特征图平均成一个特征 也是展平的效果
        x = GlobalAveragePooling2D()(x)

        # 全连接网络
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.1)(x)
        x = Dense(self.n_class, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        # model.summary()
        return model

    def predict(self, img):
        x = img
        x = cv2.resize(x, (self.sizew, self.sizeh))  # (width, height)
        X = [x]
        X = np.array(X)
        X = X / 255.0
        y = self.model.predict(X)  # 格式为[[0.96, 0.04]]
        y_label = np.argmax(y)  # 转化为0或1，0为倒，1为正
        return y_label


class CRNN(object):
    """
    CRNN模型
    """
    def __init__(self, model_path, charfile_path):
        self.img_h = 32
        self.img_w = 240
        self.batch_size = 2
        self.maxlabellength = 10
        self.model_path = model_path
        self.charfile_path = charfile_path
        self.n_class, self.char_set = self.get_n_class()
        self.basemodel = self.creat_model()
        print('{} model loaded.'.format(self.model_path))

    def creat_model(self):
        input = Input(shape=(self.img_h, None, 1), name='the_input')
        y_pred = densenet.crnn2(input, self.n_class)

        basemodel = Model(inputs=input, outputs=y_pred)
        # basemodel.summary()

        labels = Input(name='the_labels', shape=[None], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])

        model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
        # model.summary()
        model.load_weights(self.model_path)
        return basemodel

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def get_n_class(self):
        char_set = open(self.charfile_path, 'r', encoding='utf-8').readlines()
        char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])  #
        return len(char_set), char_set

    def decode(self, pred):
        '''
        ctc模型输出decode编码为最终输出
        :param pred: 模型输出
        :return:
        '''
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.n_class - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(self.char_set[pred_text[i]])
        return u''.join(char_list)

    def predict(self, img):
        out = ''
        try:
            # cv2.imshow('',img)
            # cv2.waitKey(0)
            img = Image.fromarray(img).convert('L')
            width, height = img.size[0], img.size[1]
            # if height < 20:
            #     return ''
            scale = height * 1.0 / self.img_h
            width = int(width / scale)
            img = img.resize([width, self.img_h], Image.ANTIALIAS)
            # cv2.imwrite('save_2_' + img_name, np.array(img))
            img = np.array(img).astype(np.float32) / 255.0 - 0.5
            x = img.reshape([1, self.img_h, width, 1])
            if width > 15:
                y_pred = self.basemodel.predict(x)
                # print(y_pred)
                y_pred = y_pred[:, :, :]  # batch_size,time_step,num_classes
                out = self.decode(y_pred)
            # print(out)
        except:
            pass
        return out

    def crop_predict(self, img, rec):
        crop = img[rec[1]:rec[3], rec[0]:rec[2]]
        out = self.predict(crop)
        return out

if __name__ == '__main__':
    # data_path = 'imgs'      # 数据集路径
    data_path = 'E:/projects/python/DF_idcard/data/Test'
    # data_path = 'E:/projects/python/DF_idcard/data/train_data'

    output_path = 'output_imgs'  # 保存中间文件的路径
    notrect_file = output_path + "/notrec_imgs.csv"  # 保存为进行识别的图片名

    yolo = YOLO()    # 加载yolo模型
    unet = Unet()    # 加载Unet模型
    # ctpn = CPTN()     # 加载CPTN模型

    # 加载分类模型
    # front_weights_path = 'model/front_classify_weights.h5'     # 正面分类权值文件
    front_weights_path = 'model/0904_zf_front_weights-net-0010-0.000008.h5'  # 正面分类权值文件

    back_weights_path = 'model/back_classify_weights.h5'    # 背面分类权值文件
    front_classify = Classify(front_weights_path)
    back_classify = Classify(back_weights_path)

    # 加载CRNN模型
    # char_weighs_path = 'model/char.h5'
    # letternum_weights_path = 'model/letternum.h5'
    # char_file_path = 'model_data/char.txt'
    # letternum_file_path = 'model_data/letternum.txt'
    # char_crnn = CRNN(char_weighs_path, char_file_path)
    # letternum_crnn = CRNN(letternum_weights_path, letternum_file_path)

    path_list = os.listdir(data_path)  # 列出文件夹下所有的目录与文件

    # 建立result.csv提交文件
    path = output_path + '/result.csv'
    with open(path, 'wb') as f:
        csv_write = csv.writer(f)
        pass

    yolo_list = []

    # 循环进行识别
    for i in range(0, len(path_list)):
        name, ext = os.path.splitext(path_list[i])
        if ext == '.png' or ext == '.PNG' or ext == '.JPG' or ext == '.jpg':
            print(path_list[i])
            img_path = data_path + '/' + path_list[i]
            img_name = img_path.split('/')[-1]

            # yolo框出正面背面图片
            image = Image.open(img_path)
            rects, classes = yolo.detect_image(image)  # <class 'list'>: [{'score': '0.064195335', 'x1': '94.0', 'y1': '492.0', 'x2': '635.0', 'y2': '815.0'}, {'score': '0.061926477', 'x1': '78.0', 'y1': '82.0', 'x2': '660.0', 'y2': '390.0'}], #  class 'list'>: ['fan', 'zheng']

            if (len(classes)!=2 or 'zheng' not in classes or 'fan' not in classes):
                with open(notrect_file, 'a+') as f:
                    print('error =========> ',img_path,len(classes))
                    f.write(img_name +","+ "+".join(classes) + '\n')
                # continue

            if len(set(classes)) != 2:
                continue

            card_info = {}
            # 抠图：用yolo得到的坐标在unet还原后的图片上抠图
            box_list = []
            merge_blank = 0  # 两幅图接近时有问题，暂时删除padding

            for idx, rec in enumerate(rects):

                if classes[idx] == 'zheng':

                    if card_info.__contains__('front_score') and float(card_info['front_score']) > float(rec['score']):
                        continue

                    card_info['file'] = name
                    card_info['front_score'] = float(rec['score'])
                    card_info['front_x1'] = int(float(rec['x1']))
                    card_info['front_y1'] = int(float(rec['y1']))
                    card_info['front_x2'] = int(float(rec['x2']))
                    card_info['front_y2'] = int(float(rec['y2']))

                    box_list.append(
                        [int(float(rec['x1'])), max(int(float(rec['y1'])) - merge_blank, 0),
                         int(float(rec['x2'])), min(int(float(rec['y2'])) + merge_blank, 1000)])  # objBoxSet <class 'list'>: [['94.0', '492.0', '635.0', '815.0'], ['78.0', '82.0', '660.0', '390.0']]

                if classes[idx] == 'fan':

                    if card_info.__contains__('back_score') and float(card_info['back_score']) > float(rec['score']):
                        continue

                    card_info['file'] = name
                    card_info['back_score'] = float(rec['score'])
                    card_info['back_x1'] = int(float(rec['x1']))
                    card_info['back_y1'] = int(float(rec['y1']))
                    card_info['back_x2'] = int(float(rec['x2']))
                    card_info['back_y2'] = int(float(rec['y2']))

                    box_list.append(
                        [int(float(rec['x1'])), max(int(float(rec['y1'])) - merge_blank, 0),
                         int(float(rec['x2'])), min(int(float(rec['y2'])) + merge_blank, 1000)])  # objBoxSet <class 'list'>: [['94.0', '492.0', '635.0', '815.0'], ['78.0', '82.0', '660.0', '390.0']]

            # 新增加 计算两幅接近程度 1017
            min_space = min(abs(float(card_info['back_y1']) - float(card_info['front_y2'])),abs(float(card_info['front_y1']) - float(card_info['back_y2'])))

            # 寻找图片边界
            back_front_flag = 0
            merge_blank_min_h = 15 # 最小边界
            merge_blank_nor_h = 15 # 正常边界
            if min_space < 30:
                merge_blank_min_h = int(min_space * 0.5)  # 收缩图片

                if min_space < 15:
                    # merge_blank_min_h = -3 # 收缩图片,防止重叠
                    merge_blank_min_h = 0  # 收缩图片,防止重叠

                # back - front
                if abs(float(card_info['back_y1']) - float(card_info['front_y2'])) < abs(float(card_info['front_y1']) - float(card_info['back_y2'])):
                    back_front_flag = 1 # 背面在上，正面在下
            else:
                merge_blank_min_h = 15  # 增加边界

            # # --------------------原图（保存在不同的文件夹） 尝试 会导致（f940a20e64dd42ddaf157ae830d9e1d0）颠倒（71efe9b12bb44f2f96e3e0adb80304a7）倾斜
            # img_leave = cv2.imread(img_path)
            img_src = cv2.imread(img_path)

            # unet对原图去水印
            y = unet.predict(img_path)
            cv2.imwrite(output_path + '/unet_img.jpg', y)
            img_leave = cv2.imread(output_path + '/unet_img.jpg')

            croped_dict = {}  # 正面图片为img_crops[0], 背面图片为img_crops[1]
            src_croped_dict = {}

            for idx, box in enumerate(box_list):
                # 遍历每种类型的锚点框
                img_class = classes[idx]

                top_blank = merge_blank_nor_h
                bottom_blank = merge_blank_nor_h

                # 存在不同的边界高度(两幅图 中间部分
                if merge_blank_min_h != merge_blank_nor_h :

                    if back_front_flag == 1 and img_class == "zheng" :
                        top_blank = merge_blank_nor_h
                        bottom_blank = merge_blank_min_h
                    else:
                        top_blank = merge_blank_min_h
                        bottom_blank = merge_blank_nor_h

                cropped = img_leave[(box[1] - top_blank):(box[3] + bottom_blank), (box[0] - merge_blank_nor_h):(box[2] + merge_blank_nor_h)]  # 裁剪坐标为[y0:y1, x0:x1]
                croped_dict[img_class] = cropped

                cropped = img_src[(box[1] - top_blank):(box[3] + bottom_blank), (box[0]- merge_blank_nor_h):(box[2] + merge_blank_nor_h)]  # 裁剪坐标为[y0:y1, x0:x1]
                src_croped_dict[img_class] = cropped

                # 调整后坐标保存
                if classes[idx] == 'zheng':
                    card_info['front_x1'] = int(card_info['front_x1']) - merge_blank_nor_h
                    card_info['front_y1'] = int(card_info['front_y1']) - top_blank
                    card_info['front_x2'] = int(card_info['front_x2']) + merge_blank_nor_h
                    card_info['front_y2'] = int(card_info['front_y2']) + bottom_blank

                if classes[idx] == 'fan':
                    card_info['back_x1'] = int(card_info['back_x1']) - merge_blank_nor_h
                    card_info['back_y1'] = int(card_info['back_y1']) - top_blank
                    card_info['back_x2'] = int(card_info['back_x2']) + merge_blank_nor_h
                    card_info['back_y2'] = int(card_info['back_y2']) + bottom_blank

            # front_img = croped_dict['zheng']
            # back_img = croped_dict['fan']
            front_img = src_croped_dict['zheng']
            back_img = src_croped_dict['fan']

            cv2.imwrite(output_path + '/front.jpg', front_img)
            cv2.imwrite(output_path + '/back.jpg', back_img)

            # 图像正面反面的正倒分类
            front_y = front_classify.predict(src_croped_dict['zheng'])    # 分类标签：0为倒，1为正
            back_y = back_classify.predict(croped_dict['fan'])

            front_rotate_angle = 0
            back_rotate_angle = 0

            # 图像旋转+校正
            front_process_img = np.ones((1000, 1000, 3))
            back_process_img = np.ones((1000, 1000, 3))
            if (front_y == 0):
                front_img_step1,front_rotate_angle = jiaozheng(front_img)
                cv2.imwrite(output_path+'/front_img_step1.jpg', front_img_step1)
                front_img_step2 = rotate_bound(front_img_step1, 180)
                front_process_img = front_img_step2
            else:
                cv2.imwrite(output_path+'/front_img_step1.jpg', front_img)
                front_img_step2, front_rotate_angle = jiaozheng(front_img)
                front_process_img = front_img_step2
            if (back_y == 0):
                # back_img_step1, back_rotate_angle = jiaozheng(back_img, 150)
                back_img_step1,back_rotate_angle = jiaozheng(back_img,120) # 训练集有错误图 150->120
                cv2.imwrite(output_path+'/back_img_step1.jpg', back_img_step1)
                back_img_step2 = rotate_bound(back_img_step1, 180)
                back_process_img = back_img_step2
            else:
                cv2.imwrite(output_path+'/back_img_step1.jpg', back_img)
                # back_img_step2, back_rotate_angle = jiaozheng(back_img,150)
                back_img_step2, back_rotate_angle = jiaozheng(back_img, 120) # 训练集有错误图 150->120
                back_process_img = back_img_step2

            cv2.imwrite(output_path + '/jz_front.jpg', front_process_img)
            cv2.imwrite(output_path + '/jz_back.jpg', back_process_img)

            card_info['front_rotate_180'] = front_y
            card_info['back_rotate_180'] = back_y

            card_info['front_rotate_angle'] = front_rotate_angle
            card_info['back_rotate_angle'] = back_rotate_angle

            yolo_list.extend([card_info])

            print('i=',i)


    detect_pd = pd.DataFrame(yolo_list)
    # detect_pd.to_csv(output_path +'/yolo_rect_train_1017.csv',index= False)
    detect_pd.to_csv(output_path + '/yolo_rect_test_1017.csv', index=False)

    # yolo.close_session()
