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
PART_ADDR_PATH = '../ext_data/corpus/nbs_city_district_word_entry_for_train.txt'

# 保存生成的训练文件的位置
DATA_PATH = 'D:/projects/data/ccf2019_ocr/office_l2'

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
text_loc['birthday_y'] = (92, 113)  # 生日（年）
text_loc['birthday_m'] = (155, 113)  # 生日（月）
text_loc['birthday_d'] = (204, 113)  # 生日（日）
text_loc['addr'] = (89, 145)  # 地址 每行12个字

# 反面
text_loc['office'] = (175, 198)  # 发证机关
text_loc['valid'] = (175, 234)  # 有效期

# 出生年月日列表
birthday_year = [str(i) for i in list(range(1900, 2020))]
birthday_month = [str(i) for i in list(range(1, 13))]
birthday_day = [str(i) for i in list(range(1, 32))]

# 中国地址
# address_char = pd.read_csv('../ext_data/corpus/char(nbs_id_card_code_ver1_l10).txt',header=None)
# address_list = address_char.iloc[:, 0].tolist()
address_list = pd.read_csv(PART_ADDR_PATH,header=None).iloc[:, 0].tolist()

stamp_list = ['stamp1', 'stamp11', 'stamp12', 'stamp13', 'stamp14', 'stamp15', 'stamp16', 'stamp17', 'stamp18']

def random_rotate(min_rate=0, max_rate=5):
    rotate_rate = random.randint(min_rate, max_rate)
    if random.randint(0, 1):
        rotate_rate *= -1
    return rotate_rate

def random_stamp_loc( base_location):
    # (180, 50) 印章的区域
    loc_x = random.randint(10, IMAGE_WIDTH - 180)
    loc_y = random.randint(10, IMAGE_HIGHT - 50)
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

def get_name_image(image_w = 110,image_h = 40):
    card_info = {}
    # address_len = random.randint(1,2)
    text_len = 2
    card_info['uuid'] = uuid.uuid4().hex
    # 分局、公安局和空随机（防止过拟合）
    card_info['office'] = "".join(random.sample(address_list, text_len)) + random.choice(['分局','公安局','公安',''])

    # 生日（不考虑日期有效性）
    card_info['birthday_y'] = random.choice(birthday_year)
    # card_info['birthday_m'] = random.choice(birthday_month)
    # card_info['birthday_d'] = random.choice(birthday_day)

    # 有效期（开始日）
    card_info['valid_start'] = str(
        random.choice(birthday_year[int(card_info['birthday_y']) - 1900:])) \
                               + "." + random.choice(birthday_month).zfill(2) \
                               + "." + random.choice(birthday_day).zfill(2)

    # 有效期（截止日 一定概率设定长期、不遵循年龄）
    if random.random() < 0.8:
        card_info['valid_end'] = str(random.choice(birthday_year[int(card_info['valid_start'][:4]) - 1900:])) + \
                                 card_info['valid_start'][4:]
    else:
        card_info['valid_end'] = '长期'

    # 有效期限（显示）
    card_info['valid'] = card_info['valid_start'] + "-" + card_info['valid_end']

    # ------------------------随机生成 位置信息（角度）---------------------------
    # 随机透明化(0.4~0.85)
    card_info['back_stamp_alpha'] = random.randint(51, 216)
    # 旋转减低(模拟偏差）
    card_info['back_angle'] = random_rotate(max_rate=1)

    # 填充文字
    front_bg_img = Image.fromarray(pre_image['back'])
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
    for col in ['valid']:
        draw.text(text_loc[col], card_info[col], (font_color, font_color, font_color), font=font)

    # 多行文字填充 每行12个字
    max_line = 12
    text_line = []
    for col in ['office']:
        multi_text = ""
        for i in range(math.ceil(len(card_info[col]) / max_line)):
            text_line.append([card_info[col][i * max_line:(i + 1) * max_line]])
            multi_text += card_info[col][i * max_line:(i + 1) * max_line] + '\n'
        draw.multiline_text(text_loc[col], multi_text, (font_color, font_color, font_color), font=font, spacing=4, align='left')

    # 印章图片
    front_stamp_name = random.choice(stamp_list)
    sign_stamp(target, Image.fromarray(pre_image[front_stamp_name]).copy(), 0, (0, 0), card_info['back_stamp_alpha'], 0)

    target = target.rotate(card_info['back_angle'], Image.BICUBIC, expand=0)

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

    # # 住址
    # addr_img = main_front_img[130: 130 + 105
    # , 68: 68 + 250]
    # return target.convert('L').crop([68 + pand_w, 130 + pand_h, 68 + 250 + pand_w, 130 + 105 + pand_h]),text_line,card_info['uuid']

    # # 发证机关
    # office_img = main_back_img[192: 192 + 52
    # , 165: 165 + 240]
    # 增加的10为了防止切除黑边
    return target.convert('L').crop([165 + pand_w, 192 + pand_h, 165 + 240 + pand_w + 20, 192 + 52 + pand_h + 20]),text_line,card_info['uuid']

def ThreadCardMaker(maker_count = 100, t_name="t1"):
    '''
    多线程批量生成文件数量
    :param maker_count: 文件数据量
    :param t_name:线程名称
    :return:无
    '''

    # 标签文件夹
    file = open(DATA_PATH + "/out_txt/office_lable_by_maker_{0}.txt".format(t_name), 'w', encoding='utf-8')
    text_panding = 1

    for idx in range(maker_count):
        img,text_line,file_name = get_name_image()
        for i,text in enumerate(text_line):
            line_width = 240
            if i > 0:
                line_width = min(20+len(text[0])*25 + random.randint(1,40),line_width)
            save_file = '{0}_{1}_{2}_{3}.jpg'.format(file_name,i,len(text[0]),line_width)
            img.crop([0, 5 + 25*i - text_panding, line_width, (5 + 25*(i+1)) + text_panding]).save(DATA_PATH + '/out_img/{0}'.format(save_file))
            file.write("{0} {1}".format(save_file[:-4], text[0]) + "\n")

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
        file.flush()
    file.close()
    print('MergeTempCsv finish')

if __name__ == '__main__':

    # 计算每个线程 生成的图片数量
    W = 10000
    thread_make_count = 400 * W

    start = time.process_time()

    thread_list = []
    for i in range(2):
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






    # # 创建发证机关文件夹
    # if not os.path.exists("address_lable_by_maker.txt"):
    #     file = open('address_lable_by_maker.txt', 'w')
    #     file.write("\n")
    # else:
    #     file = open('address_lable_by_maker.txt', 'a')
    #
    # text_panding = 1
    # for i in range(10):
    #     img,adddress_line,file_name = get_name_image()
    #     for i,addr in enumerate(adddress_line):
    #         line_width = 250
    #         if i > 0:
    #             line_width = min(20+len(addr[0])*25 + random.randint(1,40),250)
    #         save_file = '{0}_{1}_{2}_{3}.jpg'.format(file_name,i,len(addr[0]),line_width)
    #         img.crop([0, 12 + 25*i - text_panding, line_width, (12 + 25*(i+1)) + text_panding]).save('out_img/{0}'.format(save_file))
    #         file.write("{0} {1}".format(save_file, addr[0]) + "\n")
    #         print("{0} {1}".format(save_file, addr[0]))
    # file.close()
