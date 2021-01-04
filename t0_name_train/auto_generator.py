import numpy as np
import keras
import os

from PIL import Image, ImageDraw, ImageFont,ImageFilter,ImageEnhance
import pandas as pd
import numpy as np
import random
import math
import uuid

IMAGE_WIDTH = 448  # 背景图片
IMAGE_HIGHT = 282  # 背景图片

IMAGE_FOLDER = '../ext_data/gen_data'

# 定数声明
# word_css = "{0}/font.TTF".format(image_folder)  # 默认字体
# word_css = "{0}/simhei.ttf".format(image_folder)  # 默认字体
word_css = "{0}/WeiRuanYaHei-1.ttf".format(IMAGE_FOLDER) # 默认字体

# df_train_image = 'd:/projects/python/DF_IdCard_Ocr/temp/name'

# 正面
text_loc={}
text_loc['name'] = (92, 47)  # 姓名
text_loc['sex'] = (92, 79)  # 性别
text_loc['nationality'] = (192, 79)  # 民族
text_loc['birthday_y'] = (92, 113)  # 生日（年）
text_loc['birthday_m'] = (155, 113)  # 生日（月）
text_loc['birthday_d'] = (204, 113)  # 生日（日）
text_loc['addr'] = (89, 145)  # 地址 每行12个字

stamp_list = ['stamp1', 'stamp11', 'stamp12', 'stamp13', 'stamp14', 'stamp15', 'stamp16', 'stamp17', 'stamp18']

def random_rotate(min_rate=0, max_rate=25):
    rotate_rate = random.randint(min_rate, max_rate)
    if random.randint(0, 1):
        rotate_rate *= -1
    return rotate_rate

def random_stamp_loc( base_location):
    # (180, 50) 印章的区域
    loc_x = random.randint(10, IMAGE_WIDTH - 180)
    loc_y = random.randint(10, IMAGE_HIGHT - 50)
    return (base_location[0] + loc_x, base_location[1] + loc_y)

class AutoImageDataGenerator(keras.utils.Sequence):

    front_image_arr = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'front_448')))

    pre_image = {}
    pre_image['stamp1'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1')))  # 复印无效
    pre_image['stamp11'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-1')))  # 复印无效
    pre_image['stamp12'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-2')))  # 复印无效
    pre_image['stamp13'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-3')))  # 复印无效
    pre_image['stamp14'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-4')))  # 复印无效
    pre_image['stamp15'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-5')))  # 复印无效
    pre_image['stamp16'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-6')))  # 复印无效
    pre_image['stamp17'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-7')))  # 复印无效
    pre_image['stamp18'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'stamp1-8')))  # 复印无效

    def sign_stamp(self,img, stamp, rotate180, location, alpha_random, gaussian_blur):
        '''
        设置印章
        :param img:
        :param stamp:
        :param rotate180:
        :param location:
        :return: 印章位置和大小
        '''

        # # 图片平滑滤波
        # stamp = stamp.filter(ImageFilter.SMOOTH)

        # # 随机 高斯模糊滤波(0.2~1.2)
        # stamp = stamp.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

        # # 印泥膨胀
        # if random.random() < 0.3:
        # 膨胀
        # stamp = stamp.filter(ImageFilter.MaxFilter(3))

        r, g, b, alpha = stamp.split()

        # 随机透明化(0.2~0.9)
        # alpha_random = random.randint(76, 230)

        alpha = alpha.point(lambda i: i > 0 and alpha_random)
        stamp.putalpha(alpha)

        # 随机旋转
        stamp = stamp.rotate(random_rotate() + rotate180, Image.BICUBIC, expand=1)

        # 记录印章位置和旋转后大小(名字概率尽可能的高）
        stamp_loc = random_stamp_loc(location)
        stamp_size = stamp.size

        # img.paste(stamp, self.random_stamp_loc(location), stamp)
        img.paste(stamp, stamp_loc, stamp)
        return stamp_loc, stamp_size

    def get_df_train_image(self, image_h=40, image_w=110, idx=0):

        select_idx = random.choice(self.df_train_index)
        pre_img = random.choice(['', 's_'])

        img_path = self.df_train_image_path[select_idx]
        label = self.df_train_user_name[select_idx]
        imgpath = os.path.join(self.df_train_dir, pre_img+img_path) + '.jpg'

        img1 = Image.open(imgpath).convert('L')

        return img1.resize((image_w, image_h), Image.ANTIALIAS), label

    def get_name_image(self, need_stamp=False, image_h=40, image_w=110 ,idx = 0, font = None):
        card_info = {}
        name_len = random.choice([1, 1, 2, 2, 2, 3])
        card_info['uuid'] = uuid.uuid4().hex

        # if idx % 2 == 0:
        #     first_name = random.choice(self.df_train_first_name_list)
        #     last_name = random.choices(self.df_train_middle_last_name_list, k = name_len)
        # else:
        first_name = random.choice(self.corpus_list)
        last_name = random.choices(self.corpus_list, k=name_len)

        card_info['name'] = " ".join([first_name] + last_name)

        # ------------------------随机生成 位置信息（角度）---------------------------
        # 增加印章的旋转限制 (原来使用默认值）
        card_info['front_stamp_angle'] = random_rotate(min_rate=0, max_rate=2)
        # 随机透明化(0.4~0.85)
        card_info['front_stamp_alpha'] = random.randint(51, 216)
        # 旋转减低(模拟偏差）
        card_info['front_angle'] = random_rotate(max_rate=2)

        # 填充文字
        front_bg_img = Image.fromarray(AutoImageDataGenerator.front_image_arr)
        target = Image.new('RGBA', (front_bg_img.size[0], front_bg_img.size[1]), (0, 0, 0, 0))
        target.paste(front_bg_img, (0, 0, front_bg_img.size[0], front_bg_img.size[1]))
        draw = ImageDraw.Draw(target)

        # 随机字的颜色（24-》30）
        font_color = random.randint(0, 30)
        card_info['font_color'] = font_color

        # 文字填充
        for col in ['name']:
            draw.text(text_loc[col], card_info[col], (font_color, font_color, font_color), font=font)

        if need_stamp == True:
            # 印章图片
            front_stamp_name = random.choice(stamp_list)
            self.sign_stamp(target, Image.fromarray(AutoImageDataGenerator.pre_image[front_stamp_name]).copy(), 0, (0, 0), card_info['front_stamp_alpha'], 0)

        # 可以试试BICUBIC -> ANTIALIAS
        target = target.rotate(card_info['front_angle'], Image.BICUBIC, expand=0)

        # 随机 图片亮度(80~120)
        image_brightness_rate = random.randint(60, 108) / 100
        enh_col = ImageEnhance.Brightness(target)
        target = enh_col.enhance(image_brightness_rate)

        # # 随机 高斯模糊滤波(0.2~1.3)
        card_info['gaussian_blur_radius'] = 0
        if random.random() < 0.9:
            # card_info['gaussian_blur_radius'] = random.randint(20, 130) / 100
            # card_info['gaussian_blur_radius'] = random.choice([0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.6])
            card_info['gaussian_blur_radius'] = random.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5])

            target = target.filter(ImageFilter.GaussianBlur(radius=card_info['gaussian_blur_radius']))

        # 随机偏移
        pand_w = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, -1, -2, -3])
        pand_h = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, -1, -2, -3])

        # 姓名
        # 4-tuple defining the left, upper, right, and lower pixel
        # target.convert('L').crop([68+pand_w, 38+pand_h, 68 + 110+pand_w, 38 + 40 + pand_h]).save('try/{0}.jpg'.format(card_info['uuid'] ))
        # target.convert('L').save('try/{0}.jpg'.format(card_info['uuid'] ))

        img1 = target.convert('L').crop([68 + pand_w, 38 + pand_h, 68 + 110 + pand_w, 38 + 40 + pand_h])

        if (self.save_image == True and idx == 0):
            self.save_image = False
            img1.save('try/{0}.jpg'.format(uuid.uuid4().hex))

        # img1 = np.array(img1, 'f') / 255.0 - 0.5
        # del target
        return img1.resize((image_w, image_h), Image.ANTIALIAS), card_info['name'].replace(' ','')

        # return target.convert('L').crop([68 + pand_w, 38 + pand_h, 68 + 110 + pand_w, 38 + 40 + pand_h]),card_info['name'],card_info['uuid']

    def charfind(self, x):
        if x in self.char_set:
            return self.char_set.find(x)
        else:
            return 0

    'Generates data for Keras'
    def __init__(self, data_size, corpus_file, df_train_dir,
                 df_train_label, max_label_length = 4, batch_size=32, image_size=[40,110],
                 df_data_ratio = 0.1):
        'Initialization'
        self.image_size = image_size
        self.batch_size = batch_size
        self.data_size = data_size
        self.corpus_file = corpus_file
        self.max_label_length = max_label_length # 每行最多文字数

        self.save_image = False
        self.df_data_ratio = df_data_ratio

        # 中国人名(姓名常用汉字)
        chinese_name_char = pd.read_csv(corpus_file)
        self.corpus_list = chinese_name_char.iloc[:, 0].tolist()

        # # DF训练[姓]
        # train_first_name_char = pd.read_csv(df_train_first_name)
        # df_train_middle_last_name_char = pd.read_csv(df_train_middle_last_name)
        # self.df_train_first_name_list = train_first_name_char.iloc[:, 0].to_list()
        # self.df_train_middle_last_name_list = df_train_middle_last_name_char.iloc[:, 0].to_list()

        char_set = open(corpus_file, 'r', encoding='utf-8').readlines()
        char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
        self.char_set = char_set
        self.n_classes = len(char_set)

        # DF提供训练集合
        df_train_img = pd.read_csv(df_train_label,sep=" ", header=None, names=['file_path', 'user_name'])
        self.df_train_image_path = df_train_img.file_path.tolist()
        self.df_train_user_name = df_train_img.user_name.tolist()
        self.df_train_index = list(range(len(self.df_train_image_path)))
        self.df_train_dir = df_train_dir

        self.front_image = Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'front_448'))

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_size / self.batch_size))

    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        return self.__data_generation()

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        self.save_image = True

    def __data_generation(self):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        x = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 1), dtype=np.float)
        labels = np.ones([self.batch_size, self.max_label_length]) * 10000

        input_length = np.zeros([self.batch_size, 1])
        label_length = np.zeros([self.batch_size, 1])

        # 身份证字体
        # word_size = 17  # 文字大小
        font = ImageFont.truetype(word_css, random.choice([16, 17, 18]))

        # 随机抽取 df训练集20% 放入batch size内
        batch_list = list(range(self.batch_size))
        batch_list = random.sample(batch_list,int(self.batch_size * self.df_data_ratio))

        for i, j in enumerate(range(self.batch_size)):

            if i in batch_list:
                img, label = self.get_df_train_image(self.image_size[0], self.image_size[1], i)
            else:
                img, label = self.get_name_image(True, self.image_size[0], self.image_size[1], i, font)

            x[i] = np.expand_dims( np.array(img, 'f') / 255.0 - 0.5, axis=2)

            label_length[i] = len(label)

            input_length[i] = self.image_size[1] // 4 + 1

            # 将str里面的字符转化为数字标签
            labels[i, :len(label)] = [self.charfind(word) for word in label]

            if (len(label) <= 0):
                print("len < 0", j)

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return (inputs, outputs)