from PIL import Image, ImageDraw, ImageFont,ImageFilter,ImageEnhance
import pandas as pd
import numpy as np
import random
import math
import uuid

class CardMaker(object):
    '''
    生成身份证图片，需要安装PIL
    结构体定义：
        person_info: 用户信息（dict）
             'addr': '内蒙古包头市达尔罕茂明安联合旗大连路',# 身份证正面 信息 户籍地址
             'area_code': '150223',       # 身份证正面 隐藏信息 地址所在区的编码
             'back_angle': 3,             # 身份证背面 旋转角度
             'back_new_loc': (334, 626),  # 身份证背面 填充背景后 坐标
             'back_new_size': (464, 306), # 身份证背面 证件旋转后 新尺寸
             'back_rotate_180': 0,        # 身份证背面 是否翻转（180 或 0）
             'back_stamp_alpha': 169,     # 身份证背面 印章 透明度 (例如：169÷255 ≈ 66%透明度）
             'back_stamp_angle': -19,     # 身份证背面 印章 旋转角度（负值 逆时针）
             'birthday_d': '16',          # 身份证正面 信息 生日（日） 注意：不足两位时，左侧填充空白对齐
             'birthday_m': '  8',         # 身份证正面 信息 生日（月） 注意：不足两位时，左侧填充空白对齐
             'birthday_y': '1940',        # 身份证正面 信息 生日（年）
             'front_angle': -1,           # 身份证正面 旋转角度
             'front_new_loc': (339, 141), # 身份证正面 填充背景后 坐标
             'front_new_size': (454, 290),# 身份证正面 证件旋转后 新尺寸
             'front_rotate_180': 0,       # 身份证正面 是否翻转（180 或 0）
             'front_stamp_alpha': 169,    # 身份证正面 印章 透明度 (例如：169÷255 ≈ 66%透明度）
             'front_stamp_angle': -13,    # 身份证正面 印章 旋转角度（负值 逆时针）
             'brightness_rate': 0.86,     # 训练图片 亮度值（调整后的亮度）
             'gaussian_blur_radius': 0.76,# 训练图片 高斯模糊滤波 （随机值域 0.2~1.2)
             'id_code': '150223194008167001', # 身份证正面 信息 身份证号码 注意：10%概率生成17位错误号码
             'id_code_err': 0,            # 身份证正面 信息 身份证号码错误标示（0 或 1）
             'name': '朱伟豪',             # 身份证正面 信息 姓名（随机 姓名 列表）
             'nationality': '拉祜',        # 身份证正面 信息 民族（随机 民族 列表）
             'office': '包头市达尔罕茂明安联合旗公安局',# 身份证背面 信息 发证机关（根据地址自动生成）
             'province': '内蒙古',         # 身份证背面 隐藏信息 所在省
             'sex': '男',                  # 身份证正面 信息 性别（男、女）
             'valid': '1978.04.25-长期',   # 身份证背面 信息 有效期间
             'valid_end': '长期',          # 身份证背面 隐藏信息 有效开始日
             'valid_start': '1978.04.25'   # 身份证背面 隐藏信息 有效截止日
             'birth_area_code': '430623'   # 身份证正面 隐藏信息 出生地址区域（身份证前六位、模拟人口迁移 20%概率）
             'front_stamp_loc': (367, 203) # 身份证正面 印章 印章位置（x,y)
             'front_stamp_size': (180, 50) # 身份证正面 印章 旋转后印章新尺寸
             'back_stamp_loc': (186, 574)  # 身份证背面 印章 印章位置（x,y)
             'back_stamp_size': (188, 90)  # 身份证背面 印章 旋转后印章新尺寸
             'uuid':'69b9cf8245fe4b6d8f96e9664a5083bf' # 图片编号（随机）,防止身份证号码重复
             'front_stamp_name': # 身份证正面 印章 选择的印章名称
             'back_stamp_name': # 身份证正面 印章 选择的印章名称
             'image_resize_rate': # 身份证 图像 随机缩放图片尺寸比例 0.8~1.1倍
             'stamp_resize_rate': # 印章 图像 随机缩放图片尺寸比例 0.8~1.1倍
    版本：0.2 20190824 增加印章定位信息、增加变更户籍数据
    作者：Huan
    '''

    IMAGE_WIDTH = 448 # 背景图片
    IMAGE_HIGHT = 282 # 背景图片

    GB_IMAGE_WIDTH = 1000 # 训练图片
    GB_IMAGE_HIGHT = 1000 # 训练图片

    def __init__(self, image_folder='gen_data'):

        self.pre_image = {}
        self.text_loc = {}

        # 定数声明
        word_css = "{0}/WeiRuanYaHei-1.ttf".format(image_folder)  # 默认字体
        word_size = 16  # 文字大小

        # 身份证字体
        self.font = ImageFont.truetype(word_css, word_size)

        # 中国人名
        chinese_name = pd.read_csv('{0}/chinese_name.csv'.format(image_folder), encoding="gbk")
        self.name_list = chinese_name.name.tolist()
        del chinese_name

        # 中国民族
        chinese_nationality = pd.read_csv('{0}/chinese_nationality.csv'.format(image_folder), encoding="gbk")
        self.nationality_list = chinese_nationality.name.tolist()
        del chinese_nationality

        # 地址(有些道路名称，后期没有使用）
        chinese_street = pd.read_csv('{0}/chineses_street.csv'.format(image_folder), encoding="utf8")
        self.street_list = chinese_street.name.tolist()
        del chinese_street

        # 身份证列表
        id_card_pd = pd.read_csv('{0}/id_card_code.csv'.format(image_folder), encoding="gbk", dtype={'code': str})
        self.id_card_key = id_card_pd.code.values
        self.id_card_dict = dict(zip(id_card_pd.code.values, id_card_pd.name.values))
        del id_card_pd

        # 出生年月日列表
        self.birthday_year = [str(i) for i in list(range(1900, 2020))]
        self.birthday_month = [str(i) for i in list(range(1, 13))]
        self.birthday_day = [str(i) for i in list(range(1, 32))]

        # 个人号码
        self.person_cod_list = [str(i) for i in list(range(1, 1000))]

        # 验证码（第十八位）
        self.id_code_check_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'X']

        # ------------------初始化 图片 和 坐标----------------
        for num in range(10):
            self.pre_image[str(num)] = Image.open('{0}/{1}.png'.format(image_folder,num))

        self.pre_image['X'] = Image.open('{0}/{1}.png'.format(image_folder,'X'))

        self.pre_image['stamp1'] = Image.open('{0}/{1}.png'.format(image_folder,'stamp1'))  # 复印无效
        self.pre_image['stamp11'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-1'))  # 复印无效
        self.pre_image['stamp12'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-2'))  # 复印无效
        self.pre_image['stamp13'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-3'))  # 复印无效
        self.pre_image['stamp14'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-4'))  # 复印无效
        self.pre_image['stamp15'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-5'))  # 复印无效
        self.pre_image['stamp16'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-6'))  # 复印无效
        self.pre_image['stamp17'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-7'))  # 复印无效
        self.pre_image['stamp18'] = Image.open('{0}/{1}.png'.format(image_folder, 'stamp1-8'))  # 复印无效

        self.pre_image['stamp2'] = Image.open('{0}/{1}.png'.format(image_folder,'stamp2'))  # 禁止复印

        self.pre_image['front'] = Image.open('{0}/{1}.png'.format(image_folder,'front_448'))
        self.pre_image['back'] = Image.open('{0}/{1}.png'.format(image_folder,'back_448'))

        # 正面
        self.text_loc['name'] = (92, 47)  # 姓名
        self.text_loc['sex'] = (92, 79)  # 性别
        self.text_loc['nationality'] = (192, 79)  # 民族
        self.text_loc['birthday_y'] = (92, 113)  # 生日（年）
        self.text_loc['birthday_m'] = (155, 113)  # 生日（月）
        self.text_loc['birthday_d'] = (204, 113)  # 生日（日）
        self.text_loc['addr'] = (89, 145)  # 地址 每行12个字

        num_width = 13  # 图片宽度
        adj = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]  # 字符间距误差

        # 初始化 身份证号码坐标
        for i in range(18):
            self.text_loc['code{0}'.format(i)] = (135 + num_width * i + adj[i], 232)  # 身份证号码（第N位）

        # 反面
        self.text_loc['office'] = (175, CardMaker.IMAGE_HIGHT + 198)  # 发证机关
        self.text_loc['valid'] = (175, CardMaker.IMAGE_HIGHT + 234)  # 有效期

    def random_rotate(self, min_rate=0, max_rate=25):
        rotate_rate = random.randint(min_rate, max_rate)
        if random.randint(0,1):
            rotate_rate *= -1
        return rotate_rate

    def readom_resize(self, min_rate=0.8, max_rate=1.1):
        resize_rate = random.uniform(min_rate, max_rate)
        if random.randint(0,1):
            resize_rate = 1
        return resize_rate

    def random_stamp_loc(self, base_location):
        # (180, 50) 印章的区域
        loc_x = random.randint(10, self.IMAGE_WIDTH - 180)
        loc_y = random.randint(50, self.IMAGE_HIGHT - 50)
        return (base_location[0] + loc_x, base_location[1] + loc_y)

    def random_card_info(self):
        '''
        随机生成用户信息
        :return:
        '''
        card_info = {}

        # ------------------------随机生成 显示信息---------------------------
        card_info['uuid'] = uuid.uuid4().hex
        card_info['name'] = random.choice(self.name_list)
        card_info['sex'] = random.choice(['男', '女'])
        card_info['nationality'] = random.choice(self.nationality_list)

        # 生日（不考虑日期有效性）
        card_info['birthday_y'] = random.choice(self.birthday_year)
        card_info['birthday_m'] = random.choice(self.birthday_month)
        card_info['birthday_d'] = random.choice(self.birthday_day)

        # 证件区域（身份证前六位，当前户籍地）
        card_info['area_code'] = random.choice(self.id_card_key)

        # 证件区域（原始户籍的身份证前六位，模拟人口迁移 20%）
        if random.random() < 0.2:
            card_info['birth_area_code'] = random.choice(self.id_card_key)
        else:
            card_info['birth_area_code'] = card_info['area_code']

        # 住址（随机生成住址）
        card_info['addr'] = self.id_card_dict[card_info['area_code']] + random.choice(self.street_list)

        # 模拟缺损ID号（10%的概率）
        if random.random() < 0.9:
            card_info['id_code_err'] = 0
            card_info['id_code'] = card_info['birth_area_code'] + str(card_info['birthday_y']) + card_info[
                'birthday_m'].zfill(2) + card_info['birthday_d'].zfill(2) \
                                   + random.choice(self.person_cod_list).zfill(3) + random.choice(self.id_code_check_list)
        else:
            card_info['id_code_err'] = 1  # 身份号码位数不足
            card_info['id_code'] = card_info['birth_area_code'] + str(card_info['birthday_y']) + card_info[
                'birthday_m'].zfill(2) + card_info['birthday_d'].zfill(2) \
                                   + random.choice(self.person_cod_list).zfill(3)

        # 所在省市（剔除发证机关）
        card_info['province'] = self.id_card_dict[card_info['area_code'][:2] + "0000"]

        # 发证机关
        card_info['office'] = self.id_card_dict[card_info['area_code']][len(card_info['province']):] + "公安局"

        # 有效期（开始日）
        card_info['valid_start'] = str(
            random.choice(self.birthday_year[int(card_info['birthday_y']) - 1900:])) \
                                   + "." + random.choice(self.birthday_month).zfill(2) \
                                   + "." + random.choice(self.birthday_day).zfill(2)

        # 有效期（截止日 一定概率设定长期、不遵循年龄）
        if random.random() < 0.8:
            card_info['valid_end'] = str(random.choice(self.birthday_year[int(card_info['valid_start'][:4]) - 1900:])) + \
                                     card_info['valid_start'][4:]
        else:
            card_info['valid_end'] = '长期'

        # 有效期限（显示）
        card_info['valid'] = card_info['valid_start'] + "-" + card_info['valid_end']

        # ------------------------随机生成 位置信息（角度）---------------------------
        # 0905 增加印章的旋转限制 (原来使用默认值）
        card_info['front_stamp_angle'] = self.random_rotate(min_rate=0, max_rate=10)
        card_info['back_stamp_angle'] = self.random_rotate(min_rate=0, max_rate=10)

        # 随机透明化(0.4~0.85)
        card_info['front_stamp_alpha'] = random.randint(51, 216)
        card_info['back_stamp_alpha'] = random.randint(51, 216)

        # 卡片旋转
        if random.random() < 0.8:
            front_rotate_180 = 0
        else:
            front_rotate_180 = 180

        if random.random() < 0.8:
            back_rotate_180 = 0
        else:
            back_rotate_180 = 180

        card_info['front_rotate_180'] = front_rotate_180
        card_info['back_rotate_180'] = back_rotate_180

        # 旋转减低 15 变10
        card_info['front_angle'] = self.random_rotate(max_rate=10) + front_rotate_180
        card_info['back_angle'] = self.random_rotate(max_rate=10) + back_rotate_180

        # 印章图片 随机放大或缩小（等比例、正反两面相同比例）
        # 随机缩放减少
        card_info['image_resize_rate'] = self.readom_resize(min_rate=0.97,max_rate=1.03)
        card_info['stamp_resize_rate'] = self.readom_resize()

        return card_info

    def sign_stamp(self, img, stamp, rotate180, location, alpha_random, gaussian_blur):
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
        stamp = stamp.rotate(self.random_rotate() + rotate180, Image.BICUBIC, expand=1)

        # 记录印章位置和旋转后大小
        stamp_loc = self.random_stamp_loc(location)
        stamp_size = stamp.size

        # img.paste(stamp, self.random_stamp_loc(location), stamp)
        img.paste(stamp, stamp_loc, stamp)
        return stamp_loc, stamp_size

    def card_maker(self, card_info):
        '''
        用于生成彩色的身份证图片，无背景。
        :param card_info: 用户信息
        :return: (身份证正面图片，身份证反面图片）
        '''

        front_bg_img = self.pre_image['front']
        back_bg_img = self.pre_image['back']

        target = Image.new('RGBA', (front_bg_img.size[0], front_bg_img.size[1] * 2), (0, 0, 0, 0))
        #     bg_img = bg_img.convert("RGBA") # 不确定是否需要？

        # 创建画板（正面、反面）
        target.paste(front_bg_img, (0, 0, front_bg_img.size[0], front_bg_img.size[1]))
        target.paste(back_bg_img, (0, back_bg_img.size[1], back_bg_img.size[0], back_bg_img.size[1] * 2))

        # 月、日 补空白
        if len(card_info['birthday_m']) == 1:
            card_info['birthday_m'] = '  ' + card_info['birthday_m']

        if len(card_info['birthday_d']) == 1:
            card_info['birthday_d'] = '  ' + card_info['birthday_d']

        # 填充文字
        draw = ImageDraw.Draw(target)

        font_color = random.randint(0, 24)
        card_info['font_color'] = font_color

        # 文字填充
        for col in ['name', 'sex', 'nationality', 'birthday_y', 'birthday_m', 'birthday_d', 'valid']:
            draw.text(self.text_loc[col], card_info[col], (font_color, font_color, font_color), font=self.font)

        # 身份证号码
        for index, num in enumerate(card_info['id_code']):
            num_img = self.pre_image[num]
            target.paste(num_img, self.text_loc['code{0}'.format(index)], num_img)

        # 多行文字填充 每行12个字
        max_line = 12
        for col in ['addr', 'office']:
            multi_text = ""
            for i in range(math.ceil(len(card_info[col]) / max_line)):
                multi_text += card_info[col][i * max_line:(i + 1) * max_line] + '\n'
            draw.multiline_text(self.text_loc[col], multi_text, (font_color, font_color, font_color), font=self.font, spacing=4, align='left')

        # 图片切分
        card_img_f = target.crop((0, 0, self.IMAGE_WIDTH, self.IMAGE_HIGHT))
        card_img_b = target.crop((0, self.IMAGE_HIGHT, self.IMAGE_WIDTH, self.IMAGE_HIGHT * 2))

        card_img_f = card_img_f.rotate(card_info['front_angle'], Image.BICUBIC, expand=1)
        card_img_b = card_img_b.rotate(card_info['back_angle'], Image.BICUBIC, expand=1)

        # 随机缩放（0.8~1.1倍）
        if card_info['image_resize_rate'] != 1:
            card_img_f = card_img_f.resize((int(card_img_f.size[0] * card_info['image_resize_rate']), int(card_img_f.size[1] * card_info['image_resize_rate'])))
            card_img_b = card_img_b.resize((int(card_img_b.size[0] * card_info['image_resize_rate']), int(card_img_b.size[1] * card_info['image_resize_rate'])))

        # 旋转后的新尺寸
        card_info['front_new_size'] = card_img_f.size
        card_info['back_new_size'] = card_img_b.size

        return card_img_f, card_img_b


    def train_image_maker(self, card_info, card_img_f, card_img_b):
        '''
        用于分步骤生成训练图片，用户填充背景，随机旋转、印章、灰度化
        :param card_info: 用户信息
        :param card_img_f: 身份证正面
        :param card_img_b: 身份证反面
        :return:
        '''
        # 白色背景
        target = Image.new('RGBA', (self.GB_IMAGE_WIDTH, self.GB_IMAGE_HIGHT), (255, 255, 255))

        # 随机 摆放图片（左右余白50，上下余白20）
        front_x = random.randint(50, self.GB_IMAGE_WIDTH - card_img_f.size[0] - 50)
        front_y = random.randint(20, int(self.GB_IMAGE_HIGHT * 0.5) - card_img_f.size[1])

        back_x = random.randint(50, self.GB_IMAGE_WIDTH - card_img_b.size[0] - 50)
        back_y = random.randint(front_y + card_img_f.size[1] + 10, self.GB_IMAGE_HIGHT - card_img_b.size[1] - 20)

        # 防止 图片越界
        if back_y + card_img_b.size[1] >= self.GB_IMAGE_HIGHT - 20:
            back_y = self.GB_IMAGE_HIGHT - card_img_b.size[1] - 20

        # 随机 图片亮度(80~120)
        image_brightness_rate = random.randint(102, 108) / 100
        enh_col = ImageEnhance.Brightness(card_img_f)
        card_img_f = enh_col.enhance(image_brightness_rate)
        enh_col = ImageEnhance.Brightness(card_img_b)
        card_img_b = enh_col.enhance(image_brightness_rate)

        # card_info['image_brightness_rate'] = random.randint(103,108) / 100
        # card_img_f = card_img_f.point(lambda p: p * card_info['image_brightness_rate'])
        # card_img_b = card_img_b.point(lambda p: p * card_info['image_brightness_rate'])

        target.paste(card_img_f, (front_x, front_y), card_img_f)
        target.paste(card_img_b, (back_x, back_y), card_img_b)

        card_info['front_new_loc'] = (front_x, front_y)
        card_info['back_new_loc'] = (back_x, back_y)

        # 随机 调整亮度（70->55)
        card_info['brightness_rate'] = random.randint(55, 95) / 100

        # 图片平滑滤波
        # target = target.filter(ImageFilter.SMOOTH_MORE)

        # # 随机 高斯模糊滤波(0.2~1.2)
        card_info['gaussian_blur_radius'] = 0
        if random.random() < 0.3:
            card_info['gaussian_blur_radius'] = random.randint(20, 120) / 100
        #     target = target.filter(ImageFilter.GaussianBlur(radius=card_info['gaussian_blur_radius']))

        # 原始图片
        source = target.copy()

        # 印章图片
        # stamp_list = ['stamp1', 'stamp11', 'stamp12', 'stamp13', 'stamp14','stamp15','stamp16','stamp17','stamp18','stamp19','stamp110','stamp111','stamp112','stamp113','stamp114','stamp115','stamp116','stamp117','stamp118']
        stamp_list = ['stamp1', 'stamp11', 'stamp12', 'stamp13', 'stamp14', 'stamp15', 'stamp16', 'stamp17', 'stamp18']
        front_stamp_name = random.choice(stamp_list)
        back_stamp_name = random.choice(stamp_list)

        front_stamp_loc, front_stamp_size = self.sign_stamp(target, self.pre_image[front_stamp_name].copy(), card_info['front_rotate_180'], card_info['front_new_loc'], card_info['front_stamp_alpha'],card_info['gaussian_blur_radius'])
        back_stamp_loc, back_stamp_size = self.sign_stamp(target, self.pre_image[back_stamp_name].copy(), card_info['back_rotate_180'], card_info['back_new_loc'], card_info['back_stamp_alpha'],card_info['gaussian_blur_radius'])

        card_info['front_stamp_loc'] = front_stamp_loc
        card_info['front_stamp_size'] = front_stamp_size
        card_info['back_stamp_loc'] = back_stamp_loc
        card_info['back_stamp_size'] = back_stamp_size

        card_info['front_stamp_name'] = front_stamp_name
        card_info['back_stamp_name'] = back_stamp_name

        target = target.filter(ImageFilter.GaussianBlur(radius=card_info['gaussian_blur_radius']))
        source = source.filter(ImageFilter.GaussianBlur(radius=card_info['gaussian_blur_radius']))

        # 图片平滑滤波
        target = target.filter(ImageFilter.SMOOTH_MORE)
        source = source.filter(ImageFilter.SMOOTH_MORE)

        return source.convert('L').point(lambda p: p * card_info['brightness_rate']), \
               target.convert('L').point(lambda p: p * card_info['brightness_rate'])

    def get_train_image(self):
        '''
        获取训练图片
        :return: （个人信息，无印章图片，有印章图片）
        '''

        person_info = self.random_card_info()
        image_front, image_back = self.card_maker(person_info)
        train_source_image, train_target_image = self.train_image_maker(person_info, image_front, image_back)

        del image_front
        del image_back

        return person_info, train_source_image, train_target_image

