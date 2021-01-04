import numpy as np
import keras
import os

from PIL import Image, ImageDraw, ImageFont,ImageFilter,ImageEnhance
import pandas as pd
import numpy as np
import random
import math
import uuid

class FileImageDataGenerator(keras.utils.Sequence):

    def charfind(self, x):
        if x in self.char_set:
            return self.char_set.find(x)
        else:
            return 0

    'Generates data for Keras'
    def __init__(self, data_size, corpus_file, train_dir, train_label ,df_train_dir, df_train_label ,
                 max_label_length = 12, batch_size=32, image_size=[35, 240], df_data_ratio = 0.1):

        'Initialization'
        self.image_size = image_size
        self.batch_size = batch_size
        self.data_size = data_size
        self.corpus_file = corpus_file

        # 官方训练集
        self.train_dir = train_dir
        self.df_train_dir = df_train_dir

        # 训练集合
        train_img = pd.read_csv(train_label,sep=" ", header=None, names=['file_path', 'address'])

        train_img.drop_duplicates(inplace=True)
        self.train_image = train_img.file_path.tolist()
        self.train_address = train_img.address.tolist()
        self.indexes = np.arange(len(self.train_image))

        df_train_img = pd.read_csv(df_train_label,sep=" ", header=None, names=['file_path', 'address'])
        self.df_train_image = df_train_img.file_path.tolist()
        self.df_train_address = df_train_img.address.tolist()
        self.df_indexes = np.arange(len(self.df_train_image))

        # 官方数据比例
        self.df_data_ratio = df_data_ratio

        char_set = open(corpus_file, 'r', encoding='utf-8').readlines()
        char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
        self.char_set = char_set
        self.n_classes = len(char_set)

        self.max_label_length = max_label_length # 每行最多文字数
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_size / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        batch_index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.data_generation(batch_index)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indexes)

    def data_generation(self,batch_index):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 1), dtype=np.float)
        labels = np.ones([self.batch_size, self.max_label_length]) * 10000

        input_length = np.zeros([self.batch_size, 1])
        label_length = np.zeros([self.batch_size, 1])

        # 随机抽取 df训练集20% 放入batch size内
        batch_list = list(range(self.batch_size))
        batch_list = random.sample(batch_list,int(self.batch_size * self.df_data_ratio))

        file_path = ''
        for i, idx in enumerate(batch_index):

            if i in batch_list:
                df_idx =  random.choice(self.df_indexes)
                image_name = self.df_train_image[df_idx]
                label = self.df_train_address[df_idx]
                file_path = self.df_train_dir
            else:
                image_name = self.train_image[idx]
                label = self.train_address[idx]
                file_path = self.train_dir

            image_path = os.path.join(file_path,image_name + ".jpg")
            img = Image.open(image_path).convert('L')

            # 图片缩放(width, height)
            resize_times = self.image_size[0] / img.size[1]
            img_resize = img.resize((int(img.size[0] * resize_times), self.image_size[0]), Image.ANTIALIAS)
            new_img = np.ones([self.image_size[0], self.image_size[1]]) * 255

            new_img[:img_resize.size[1], :min(img_resize.size[0], self.image_size[1])] = np.array(img_resize, 'f')[:,:min(img_resize.size[0], self.image_size[1])]
            # new_img[:img_resize.size[1],:img_resize.size[0]] = np.array(img_resize, 'f')

            x[i] = np.expand_dims(new_img / 255.0 - 0.5, axis=2)

            label_length[i] = len(label)
            input_length[i] = self.image_size[1] // 4 + 1

            # 将str里面的字符转化为数字标签
            labels[i, :len(label)] = [self.charfind(word) for word in label]

            if (len(label) <= 0):
                print("len < 0", idx)

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return (inputs, outputs)















        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        #
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
