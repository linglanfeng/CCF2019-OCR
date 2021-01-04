from keras import backend as K
import cv2
import os
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Lambda
from PIL import Image

import densenet as densenet

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
class Ctc_Decode:
    # 用tf定义一个专门ctc解码的图和会话，就不会一直增加节点了，速度快了很多
    def __init__(self ,batch_size, timestep, nclass):
        self.batch_size = batch_size
        self.timestep = timestep
        self.nclass = nclass
        self.graph_ctc = tf.Graph()
        with self.graph_ctc.as_default():
            self.y_pred_tensor = tf.placeholder(shape=[self.batch_size, self.timestep, self.nclass], dtype=tf.float32, name="y_pred_tensor")
            self._y_pred_tensor = tf.transpose(self.y_pred_tensor, perm=[1, 0, 2])  #  要把timestep 放在第一维
            self.input_length_tensor = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32, name="input_length_tensor")
            self._input_length_tensor = tf.squeeze(self.input_length_tensor, axis=1) #  传进来的是 [batch_size,1] 所以要去掉一维
            self.ctc_decode, _ = tf.nn.ctc_greedy_decoder(self._y_pred_tensor, self._input_length_tensor)
            self.decoded_sequences = tf.sparse_tensor_to_dense(self.ctc_decode[0])

            self.ctc_sess = tf.Session(graph=self.graph_ctc)

    def ctc_decode_tf(self, args):
        y_pred, input_length = args
        decoded_sequences = self.ctc_sess.run(self.decoded_sequences,
                                     feed_dict={self.y_pred_tensor: y_pred, self.input_length_tensor: input_length})
        return decoded_sequences


class CRNN(object):
    """
    CRNN模型
    """
    def __init__(self, model_path, charfile_path):
        self.img_h = 60
        self.img_w = 600
        self.batch_size = 256
        self.maxlabellength = 12
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


    def decode_ctc(self, num_result):
        # result = num_result[:, :, :]
        # in_len = np.zeros((1), dtype=np.int32)
        # in_len[0] = 150
        # r = K.ctc_decode(result, in_len, greedy=True, beam_width=1, top_paths=1)
        # r1 = K.get_value(r[0][0])
        # r1 = r1[0]
        # text = []
        # for i in r1:
        #     text.append(self.char_set[i])
        # # return r1, u''.join(text)
        # return u''.join(text)

        ctc_class = Ctc_Decode(batch_size=1, timestep=num_result.shape[1], nclass=self.n_class)
        predict_y = ctc_class.ctc_decode_tf([num_result, [[num_result.shape[1]]]])  # ctc解码

        text = []
        for i in predict_y[0]:
            text.append(self.char_set[i])
        return u''.join(text)

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

            img = Image.fromarray(img).convert('L')

            # 图片缩放(width, height)
            resize_times = self.img_h / img.size[1]
            img_resize = img.resize((int(img.size[0] * resize_times), self.img_h), Image.ANTIALIAS)
            new_img = np.ones([self.img_h, self.img_w]) * 255
            new_img[:img_resize.size[1],:img_resize.size[0]] = np.array(img_resize, 'f')

            #
            #
            # width, height = img.size[0], img.size[1]
            #
            # scale = height * 1.0 / self.img_h
            # width = int(width / scale)
            # img = img.resize([width, self.img_h], Image.ANTIALIAS)
            # # cv2.imwrite('save_2_' + img_name, np.array(img))
            img = np.array(new_img).astype(np.float32) / 255.0 - 0.5

            x = img.reshape([1, self.img_h, self.img_w, 1])
            if self.img_w > 15:
                y_pred = self.basemodel.predict(x)
                # print(y_pred)
                # print('decode_ctc is ',self.decode_ctc(y_pred))
                # print('decode is ', self.decode(y_pred[:, :, :] ))

                # y_pred = y_pred[:, :, :]  # batch_size,time_step,num_classes
                # out = self.decode(y_pred)

                # 试着解决叠字
                out = self.decode_ctc(y_pred)
            # print(out)
        except:
            pass
        return out


# 预料集合来自NBS
CHAR_FILE_PATH = '../ext_data/corpus/char(nbs_id_card_code_ver1_l10).txt'
# 训练权重保存路径
WEIGHTS_PATH = '../x_train_model/address_weights/1006_address_only_180-loss-0.104651-acc-0.9823.h5'

id_crnn = CRNN(WEIGHTS_PATH, CHAR_FILE_PATH)

# weighs_path = 'D:/projects/python/DF_IdCard_Ocr/train_model/address_weights/1001_address_only_368-loss-0.100874-acc-0.9856.h5'
# file_path = 'corpus/char(train_address_id_code_address_street).txt'



# 源路径：下面是子文件夹
src_path = "D:/projects/data/ccf2019_ocr/address_l2/test_img"
output_path = '../single_result/1006_address_only_180_1014test.csv'

def merge_result(result_df):
    # result_file_name = '0925_address_result_epoch266.csv'
    # merge_result_file_name = '0925_address_result_epoch266.csv'
    # result_path = 'D:/projects/python/DF_IdCard_Ocr/result/'

    addr_pd = result_df.copy()
    addr_pd.columns = ['file_name','addr']
    addr_pd['file'] = ""
    addr_pd['idx'] = ""
    addr_pd[['file','idx']] = addr_pd.file_name.str.split('_',expand = True)

    addr_pd.sort_values(by=['file', 'idx'], ascending=True,inplace= True)
    addr_pd['addr'] = addr_pd['addr'].fillna('')

    # #剔除无用叠字
    # for idx, row in addr_pd[addr_pd.addr.str.len() > 12].iterrows():
    #     last_char = ''
    #     new_result = []
    #     for char in row['addr']:
    #         if last_char != char:
    #             new_result.extend(char)
    #             last_char = char
    #         else:
    #             last_char = ""
    #     # addr_pd.ix[idx]['addr'] = "".join(new_result)
    #     addr_pd.iloc[idx]['addr'] = "".join(new_result)

    addr_merge_pd = addr_pd.groupby(['file']).apply(lambda x: "".join(x['addr'])).reset_index()
    addr_merge_pd.columns=['file','address']

    addr_merge_pd.to_csv(output_path, encoding='utf_8_sig', header=None, index=False)


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

# 测试样本图片路径
all_imgs = walkpath(src_path, 'jpg')    # 文件夹下所有文件
# test_imgs = [x for x in all_imgs if x.split('/')[-1][-5]=='0']
test_imgs = all_imgs

# all_imgs = walkpath(src_path, 'jpg')    # 文件夹下所有文件
# test_imgs = [src_path + '/0043bc679f49400ba16f9eb5c566c04f_0.jpg'
#              ]


predict_Y = []
img_names = []
for idx,path in enumerate(test_imgs):
    x = cv2.imread(path)
    if x is not None:
        y = id_crnn.predict(x)
        print(idx, path+' : '+ str(y))
        predict_Y.append(str(y))
        img_names.append(path.replace('\\','/').split('/')[-1][:-4])

c={"name": img_names, "predict" : predict_Y,}
import pandas as pd
df = pd.DataFrame(c)

# df.to_csv(output_path, encoding='utf_8_sig', header=None, index=False)
merge_result(df)



# for file in test_imgs:
#     img_path = file
#     img = cv2.imread(img_path)
#     y = id_crnn.predict(img)
#     print(y)
#     break
