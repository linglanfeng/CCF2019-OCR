from PIL import Image, ImageDraw, ImageFont,ImageFilter,ImageEnhance
import pandas as pd
import numpy as np
import random
import math
import uuid
import os
import time
from threading import Thread

IMAGE_WIDTH = 448  # 背景图片
IMAGE_HIGHT = 282  # 背景图片

IMAGE_FOLDER='../ext_data/gen_data'
# CHAR_FILE_PATH = '../ext_data/corpus/char(120w).txt'

# 出生年月日列表
MIN_START_YEAR = 1958
birthday_year = [str(i) for i in list(range(MIN_START_YEAR,2009))] # 1958 ~ 2008
birthday_month = [str(i) for i in list(range(1, 13))]
birthday_day = [str(i) for i in list(range(1, 32))]

# 保存生成的训练文件的位置
DATA_PATH = 'D:/projects/data/ccf2019_ocr/day_l2'

# 定数声明
# word_css = "{0}/font.TTF".format(IMAGE_FOLDER)  # 默认字体
# word_css = "{0}/simhei.ttf".format(IMAGE_FOLDER)  # 默认字体
word_css = "{0}/WeiRuanYaHei-1.ttf".format(IMAGE_FOLDER) # 默认字体

# word_size = 16  # 文字大小
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

pre_image['front'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'front_448')))
pre_image['back'] = np.array(Image.open('{0}/{1}.png'.format(IMAGE_FOLDER, 'back_448')))

# 正面
text_loc={}
text_loc['name'] = (92, 47)  # 姓名
text_loc['sex'] = (92, 79)  # 性别
text_loc['nationality'] = (192, 79)  # 民族
text_loc['birthday_y'] = (92, 110)  # 生日（年）
text_loc['birthday_m'] = (155, 110)  # 生日（月）
text_loc['birthday_d'] = (204, 110)  # 生日（日）
text_loc['addr'] = (89, 145)  # 地址 每行12个字

# # 姓名 语料
# chinese_name_char = pd.read_csv(CHAR_FILE_PATH)
# corpus_list = chinese_name_char.iloc[:, 0].tolist()

stamp_list = ['stamp1', 'stamp11', 'stamp12', 'stamp13', 'stamp14', 'stamp15', 'stamp16', 'stamp17', 'stamp18']

def random_rotate(min_rate=0, max_rate=5):
    rotate_rate = random.randint(min_rate, max_rate)
    if random.randint(0, 1):
        rotate_rate *= -1
    return rotate_rate

def random_stamp_loc( base_location):
    # (180, 50) 印章的区域
    # loc_x = random.randint(10, IMAGE_WIDTH - 180)
    # loc_y = random.randint(10, IMAGE_HIGHT - 50)

    loc_x = random.randint(10, IMAGE_WIDTH - 280) # 1013增加印章干扰概率
    loc_y = random.randint(10, IMAGE_HIGHT - 150)
    return (base_location[0] + loc_x, base_location[1] + loc_y)

def sign_stamp(img, stamp, rotate180, location, alpha_random, gaussian_blur):
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
    stamp = stamp.rotate(random_rotate(min_rate=0, max_rate=5) + rotate180, Image.BICUBIC, expand=1)

    # 记录印章位置和旋转后大小(名字概率尽可能的高）
    stamp_loc = random_stamp_loc(location)
    stamp_size = stamp.size

    # img.paste(stamp, self.random_stamp_loc(location), stamp)
    img.paste(stamp, stamp_loc, stamp)
    return stamp_loc, stamp_size

def get_nation_image(image_w = 110,image_h = 40):
    card_info = {}
    card_info['uuid'] = uuid.uuid4().hex

    # 生日（不考虑日期有效性）
    card_info['birthday_y'] = random.choice(birthday_year)
    card_info['birthday_m'] = random.choice(birthday_month)
    card_info['birthday_d'] = random.choice(birthday_day)

    # np.argwhere(card_info['birthday_y'] )

    # ------------------------随机生成 位置信息（角度）---------------------------
    # 随机透明化(0.4~0.85)
    card_info['front_stamp_alpha'] = random.randint(51, 216)
    # 旋转减低(模拟偏差）
    card_info['front_angle'] = random_rotate(max_rate=1)

    # 填充文字
    front_bg_img = Image.fromarray(pre_image['front'])
    target = Image.new('RGBA', (front_bg_img.size[0], front_bg_img.size[1]), (0, 0, 0, 0))
    target.paste(front_bg_img, (0, 0, front_bg_img.size[0], front_bg_img.size[1]))
    draw = ImageDraw.Draw(target)

    # 随机字的颜色（24-》30）
    font_color = random.randint(0, 30)
    card_info['font_color'] = font_color

    # 文字填充
    # 身份证字体
    font = ImageFont.truetype(word_css, random.choice([16, 17, 18]))

    # 文字填充
    for col in ['birthday_y','birthday_m','birthday_d']:
        draw.text(text_loc[col], card_info[col], (font_color, font_color, font_color), font=font)

    # 印章图片
    front_stamp_name = random.choice(stamp_list)
    sign_stamp(target, Image.fromarray(pre_image[front_stamp_name]).copy(), 0, (0, 0), card_info['front_stamp_alpha'], 0)

    target = target.rotate(card_info['front_angle'], Image.BICUBIC, expand=0)

    # 随机 图片亮度(80~120)
    image_brightness_rate = random.randint(60, 108) / 100
    enh_col = ImageEnhance.Brightness(target)
    target = enh_col.enhance(image_brightness_rate)

    # # 随机 高斯模糊滤波(0.2~1.3)
    card_info['gaussian_blur_radius'] = 0
    if random.random() < 0.9:
        card_info['gaussian_blur_radius'] = random.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5])
        target = target.filter(ImageFilter.GaussianBlur(radius=card_info['gaussian_blur_radius']))

    # 随机偏移
    pand_w = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 8, -1, -2, -3, -4, -6, -8])
    pand_h = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, -1, -2, -3])

    # # 年
    # year_img = main_front_img[102: 102 + 40
    # , 68:68 + 70]

    # # 月
    # month_img = main_front_img[102: 102 + 40
    # , 132: 132 + 52]

    # # 日
    # day_img = main_front_img[102: 102 + 40
    # , 182: 182 + 52]
    return target.convert('L').crop([189 + pand_w, 102 + pand_h, 189 + 52 + pand_w, 102 + 40 + pand_h]), (int(card_info['birthday_d']) - 1), card_info['uuid']

def ThreadCardMaker(maker_count = 100, t_name="t1"):
    '''
    多线程批量生成文件数量
    :param maker_count: 文件数据量
    :param t_name:线程名称
    :return:无
    '''

    # 标签文件夹
    file = open(DATA_PATH + "/out_txt/day_lable_by_maker_{0}.txt".format(t_name), 'w', encoding='utf-8')
    text_panding = 1

    for idx in range(maker_count):
        img, user_name,file_name = get_nation_image()

        save_file = '{0}.jpg'.format(file_name)
        img.save(DATA_PATH + '/out_img/{0}'.format(save_file))
        file.write("{0} {1}".format(save_file[:-4], user_name) + "\n")

        if(idx%100==0):
            print(t_name,idx,file_name)
            file.flush()

    file.close()

def MergeTempCsv():
    '''
    合并所有temp_csv下的临时文件（不删除文件，下次再生成时会清理）
    :return:
    '''

    data_file = DATA_PATH + '/train_maker_label.txt'
    if not os.path.exists(data_file):
        file = open(data_file, 'w', encoding='utf-8')
        file.write("\n")
    else:
        file = open(data_file, 'a', encoding='utf-8')

    # 合并所有生成文件
    label_txt = DATA_PATH + '/out_txt/'
    for file_name in os.listdir(label_txt):
        print('Merge File :',file_name)

        for line in open(label_txt + file_name, encoding='utf-8'):
            file.writelines(line)

    file.close()
    print('MergeTempCsv finish')

if __name__ == '__main__':

    # 计算每个线程 生成的图片数量
    W = 10000
    # thread_make_count = 100 * W
    thread_make_count = 30 * W # 1013日再增加数据试试效果

    start = time.process_time()

    thread_list = []
    for i in range(6):
        # ----------多线程 测试（文件生成）----------
        thread_maker = Thread(target=ThreadCardMaker,args=(thread_make_count, "t{0}".format(i),))
        thread_list.append(thread_maker)

    # 启动子线程
    for t in thread_list:
        t.setDaemon(True)
        t.start()

    # 等待合并csv文件
    for t in thread_list:
        t.join()

    # 准备合并文件
    MergeTempCsv()
    print('Total time(s) : ',int(time.process_time() - start))
    print('finish')