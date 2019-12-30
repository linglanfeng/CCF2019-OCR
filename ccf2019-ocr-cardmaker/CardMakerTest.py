from CardMaker import CardMaker
from threading import Thread
import os
import time
import pandas as pd
from PIL import Image
import numpy as np

# -------------------TODO------------------
# 文件保存路径（数据根路径）
# 命名规则：[身份证号码] + [UUID].jpg
# 文件夹no_stamp：无印章图片，with_stamp：有印章图片
SAVE_PATH = 'D:/projects/data/ccf2019_ocr_share/all_in_one_train' # 保存完整的图片（1000 * 1000），正反面在一起

#https://www.jianshu.com/p/3f6abf3eeba2

# 单幅图片（正反面各一张图片）
# 命名规则：[身份证号码] + [UUID] + [b/f].jpg
# 内部文件夹：no_stamp：无印章图片，with_stamp：有印章图片，temp_csv临时文件夹
# 数据文件：data_all.csv 可以多次执行程序， 结果会自动累计
CORP_SAVE_PATH = 'c:/temp/single_side_train'

THREAD_COUNT = 5 # 生产线程数量
MAKE_COUNT = 10 * THREAD_COUNT  # 生成图片数量 (没有处理余数，所以最好能被线程数整除：线程数的整数倍）
DATA_FILE_NAME = 'data_all.csv'

# -------------------不用修改---------------
NO_STAMP_DIR = 'no_stamp'
WITH_STAMP_DIR = 'with_stamp'
TEMP_CSV = 'temp_csv'
TEMP_MAX_LINE = 100
IMAGE_SAVE_QUALITY = 95
# IMAGE_SINGLE_SIDE_W = 512
# IMAGE_SINGLE_SIDE_H = 512

IMAGE_SINGLE_SIDE_W = 480
IMAGE_SINGLE_SIDE_H = 350

def ThreadCardMaker(maker_count = 100, t_name="t1"):
    '''
    多线程批量生成文件数量
    :param maker_count: 文件数据量
    :param t_name:线程名称
    :return:无
    '''
    maker = CardMaker()
    data_list = []
    save_index = 1
    # 数据切片文件路径
    temp_path = os.path.join(SAVE_PATH, TEMP_CSV)

    for index in range(maker_count):
        # ------- 分步骤生产图片-------#
        # person_info = maker.random_card_info()
        # image_front, image_back = maker.card_maker(person_info)
        # train_source_image, train_target_image = maker.train_image_maker(person_info, image_front, image_back)

        # ------- 快速生成图片（0.10s~0.15s 每张)-------#
        person_info, train_source_image, train_target_image =  maker.get_train_image()
        # 保存数据文件
        data_list.append(person_info)

        no_stamp_path = os.path.join(SAVE_PATH, NO_STAMP_DIR, '{0}_{1}.jpg'.format(person_info['id_code'], person_info['uuid']))
        with_stamp_path = os.path.join(SAVE_PATH, WITH_STAMP_DIR, '{0}_{1}.jpg'.format(person_info['id_code'], person_info['uuid']))

        # 保存图片
        train_source_image.save(no_stamp_path, quality=IMAGE_SAVE_QUALITY)
        train_target_image.save(with_stamp_path, quality=IMAGE_SAVE_QUALITY)

        #---------------整理单面图片 512 x 512
        no_stamp_path_front = os.path.join(CORP_SAVE_PATH, WITH_STAMP_DIR, 'n_{0}_{1}_f.jpg'.format(person_info['id_code'], person_info['uuid']))
        no_stamp_path_back = os.path.join(CORP_SAVE_PATH, NO_STAMP_DIR, '{0}_{1}_b.jpg'.format(person_info['id_code'], person_info['uuid']))

        with_stamp_path_front = os.path.join(CORP_SAVE_PATH, WITH_STAMP_DIR, 's_{0}_{1}_f.jpg'.format(person_info['id_code'], person_info['uuid']))
        with_stamp_path_back = os.path.join(CORP_SAVE_PATH, WITH_STAMP_DIR, '{0}_{1}_b.jpg'.format(person_info['id_code'], person_info['uuid']))

        # 裁剪（不带印章）
        no_stamp_card_front_crop = SaveSingleSide(train_source_image, person_info['front_new_loc'], person_info['front_new_size'])
        no_stamp_card_front_crop.save(no_stamp_path_front, quality=IMAGE_SAVE_QUALITY)

        # no_stamp_card_back_crop = SaveSingleSide(train_source_image, person_info['back_new_loc'], person_info['back_new_size'])
        # no_stamp_card_back_crop.save(no_stamp_path_back, quality=IMAGE_SAVE_QUALITY)

        # 裁剪(带印章）
        with_stamp_card_front_crop = SaveSingleSide(train_target_image, person_info['front_new_loc'], person_info['front_new_size'])
        with_stamp_card_front_crop.save(with_stamp_path_front, quality=IMAGE_SAVE_QUALITY)

        # with_stamp_card_back_crop = SaveSingleSide(train_target_image, person_info['back_new_loc'], person_info['back_new_size'])
        # with_stamp_card_back_crop.save(with_stamp_path_back, quality=IMAGE_SAVE_QUALITY)

        # 防止中断
        if (index+1)%TEMP_MAX_LINE == 0:
            data_pd = pd.DataFrame(data_list)
            data_pd.to_csv(temp_path+ '/{0}_{1}.csv'.format(t_name, save_index), index=False)
            data_list = []
            save_index += 1

        print(t_name, index, person_info['name'], person_info['province'],person_info['id_code'])

    # 保存最后结果
    if len(data_list) > 0:
        data_pd = pd.DataFrame(data_list)
        data_pd.to_csv(temp_path + '/{0}_{1}.csv'.format(t_name, save_index), index=False)

def SaveSingleSide(image, image_location, image_size):

    roi_box = [image_location[0], image_location[1], \
               image_location[0] + image_size[0], image_location[1] + image_size[1]]

    backgrand_color = int(np.mean(image))
    backgrand = Image.new('L', (IMAGE_SINGLE_SIDE_W, IMAGE_SINGLE_SIDE_H), backgrand_color)

    image_roi = image

    # 图片小于目标尺寸
    padding_w = 0
    padding_h = 0
    if (image_size[0] <= IMAGE_SINGLE_SIDE_W) and (image_size[1] <= IMAGE_SINGLE_SIDE_H):
        padding_w = (IMAGE_SINGLE_SIDE_W - image_size[0]) //2
        padding_h = (IMAGE_SINGLE_SIDE_H - image_size[1]) //2
        roi_box[0] = max(roi_box[0] - padding_w, 0)
        roi_box[1] = max(roi_box[1] - padding_h, 0)

        roi_box[2] = min(roi_box[2] + padding_w, 1000)
        roi_box[3] = min(roi_box[3] + padding_h, 1000)

        image_roi = image.crop(roi_box)

    # 图片大于目标区域
    if (image_size[0] > IMAGE_SINGLE_SIDE_W) or (image_size[1] > IMAGE_SINGLE_SIDE_H):

        # 等比例缩放
        scale_rate = min(IMAGE_SINGLE_SIDE_W/image_size[0], IMAGE_SINGLE_SIDE_H/image_size[1])
        # print(image_size[0],image_size[1])
        # max_side = max(image_size[0],image_size[1])
        #
        # padding_w = (max_side - image_size[0]) // 2
        # padding_h = (max_side - image_size[1]) // 2
        # roi_box[0] = max(roi_box[0] - padding_w, 0)
        # roi_box[1] = max(roi_box[1] - padding_h, 0)
        # roi_box[2] = min(roi_box[2] + padding_w, 1000)
        # roi_box[3] = min(roi_box[3] + padding_h, 1000)
        # image_roi = image.crop(roi_box).resize((IMAGE_SINGLE_SIDE_W,IMAGE_SINGLE_SIDE_H))
        image_roi = image.crop(roi_box).resize((int(round(image_size[0]*scale_rate,0)), int(round(image_size[1]*scale_rate,0))))

    backgrand.paste(image_roi, (0, 0, image_roi.size[0], image_roi.size[1]))

    return backgrand

def MergeTempCsv():
    '''
    合并所有temp_csv下的临时文件（不删除文件，下次再生成时会清理）
    :return:
    '''
    data_file = os.path.join(SAVE_PATH, DATA_FILE_NAME)
    if not os.path.exists(data_file):
        pd_all = pd.DataFrame()
    else:
        pd_all = pd.read_csv(data_file)

    # 合并所有生成文件
    for file_name in os.listdir(os.path.join(SAVE_PATH, TEMP_CSV)):
        print('Merge File :',file_name)
        file_path = os.path.join(SAVE_PATH, TEMP_CSV, file_name)
        split_pd = pd.read_csv(file_path)
        pd_all = pd.concat([pd_all, split_pd], axis=0, ignore_index=True)

    # 保存合并后的文件
    pd_all.to_csv(data_file, index=False)

    print('MergeTempCsv finish')

def CheckAndMakeDirs():

    # 数据根目录是否存在
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 单面图片（正面和反面，尺寸512 * 512）
    if not os.path.exists(CORP_SAVE_PATH):
        os.makedirs(CORP_SAVE_PATH)

    # 原始图片路径（无印章）
    no_stamp_path = os.path.join(SAVE_PATH, NO_STAMP_DIR)
    if not os.path.exists(no_stamp_path):
        os.makedirs(no_stamp_path)

    no_stamp_path = os.path.join(CORP_SAVE_PATH, NO_STAMP_DIR)
    if not os.path.exists(no_stamp_path):
        os.makedirs(no_stamp_path)

    # 带印章
    with_stamp_path = os.path.join(SAVE_PATH, WITH_STAMP_DIR)
    if not os.path.exists(with_stamp_path):
        os.makedirs(with_stamp_path)

    with_stamp_path = os.path.join(CORP_SAVE_PATH, WITH_STAMP_DIR)
    if not os.path.exists(with_stamp_path):
        os.makedirs(with_stamp_path)

    temp_path = os.path.join(SAVE_PATH, TEMP_CSV)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    else:
        # 清楚上次生产的残余文件
        for file_name in os.listdir(temp_path):
            os.remove(os.path.join(temp_path, file_name))
    # return no_stamp_path, with_stamp_path

if __name__ == '__main__':

    # ----------单线程----------
    # 创建身份证生成器
    maker = CardMaker()
    start = time.clock()

    # 无印章图片 和 带印章图片保存文件路径
    CheckAndMakeDirs()

    # 计算每个线程 生成的图片数量
    thread_make_count = MAKE_COUNT // THREAD_COUNT

    thread_list = []
    for i in range(THREAD_COUNT):

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









    #----------------------以下内容 临时保存文件 暂时不要删除u-------------------

    #
    # # 获取随机用户信息（速度估算：不保存文件 12.8s/百张， 保存文件 24s/百张)
    # for i in range(1):
    #     # ------- 快速生成图片（0.10s~0.15s/每张)-------#
    #     '''
    #     person_info: 用户信息（dict）
    #          'addr': '内蒙古包头市达尔罕茂明安联合旗大连路',# 身份证正面 信息 户籍地址
    #          'area_code': '150223',       # 身份证正面 隐藏信息 地址所在区的编码
    #          'back_angle': 3,             # 身份证背面 旋转角度
    #          'back_new_loc': (334, 626),  # 身份证背面 填充背景后 坐标
    #          'back_new_size': (464, 306), # 身份证背面 证件旋转后 新尺寸
    #          'back_rotate_180': 0,        # 身份证背面 是否翻转（180 或 0）
    #          'back_stamp_alpha': 169,     # 身份证背面 印章 透明度 (例如：169÷255 ≈ 66%透明度）
    #          'back_stamp_angle': -19,     # 身份证背面 印章 旋转角度（负值 逆时针）
    #          'birthday_d': '16',          # 身份证正面 信息 生日（日） 注意：不足两位时，左侧填充空白对齐
    #          'birthday_m': '  8',         # 身份证正面 信息 生日（月） 注意：不足两位时，左侧填充空白对齐
    #          'birthday_y': '1940',        # 身份证正面 信息 生日（年）
    #          'front_angle': -1,           # 身份证正面 旋转角度
    #          'front_new_loc': (339, 141), # 身份证正面 填充背景后 坐标
    #          'front_new_size': (454, 290),# 身份证正面 证件旋转后 新尺寸
    #          'front_rotate_180': 0,       # 身份证正面 是否翻转（180 或 0）
    #          'front_stamp_alpha': 169,    # 身份证正面 印章 透明度 (例如：169÷255 ≈ 66%透明度）
    #          'front_stamp_angle': -13,    # 身份证正面 印章 旋转角度（负值 逆时针）
    #          'brightness_rate': 0.86,     # 训练图片 亮度值（调整后的亮度）
    #          'gaussian_blur_radius': 0.76,# 训练图片 高斯模糊滤波 （随机值域 0.2~1.2)
    #          'id_code': '150223194008167001', # 身份证正面 信息 身份证号码 注意：10%概率生成17位错误号码
    #          'id_code_err': 0,            # 身份证正面 信息 身份证号码错误标示（0 或 1）
    #          'name': '朱伟豪',             # 身份证正面 信息 姓名（随机 姓名 列表）
    #          'nationality': '拉祜',        # 身份证正面 信息 民族（随机 民族 列表）
    #          'office': '包头市达尔罕茂明安联合旗公安局',# 身份证背面 信息 发证机关（根据地址自动生成）
    #          'province': '内蒙古',         # 身份证背面 隐藏信息 所在省
    #          'sex': '男',                  # 身份证正面 信息 性别（男、女）
    #          'valid': '1978.04.25-长期',   # 身份证背面 信息 有效期间
    #          'valid_end': '长期',          # 身份证背面 隐藏信息 有效开始日
    #          'valid_start': '1978.04.25'   # 身份证背面 隐藏信息 有效截止日
    #          'birth_area_code': '430623'   # 身份证正面 隐藏信息 出生地址区域（身份证前六位、模拟人口迁移 20%概率）
    #          'front_stamp_loc': (367, 203) # 身份证正面 印章 印章位置（x,y)
    #          'front_stamp_size': (180, 50) # 身份证正面 印章 旋转后印章新尺寸
    #          'back_stamp_loc': (186, 574)  # 身份证背面 印章 印章位置（x,y)
    #          'back_stamp_size': (188, 90)  # 身份证背面 印章 旋转后印章新尺寸
    #
    #     train_source_image：不含印章的图片
    #     train_target_image：包含印章的图片
    #     '''
    #     person_info, train_source_image, train_target_image = maker.get_train_image()
    #
    #     elapsed = (time.clock() - start)
    #     # print(i, elapsed, person_info['name'], person_info['province'], person_info['id_code'])
    #     print(i, elapsed, person_info)
    #
    #     # person_info['front_new_loc']
    #     #
    #     # front_box = (person_info['front_new_loc'][0], person_info['front_new_loc'][1],\
    #     #        person_info['front_new_loc'][0] + person_info['front_new_size'][0], person_info['front_new_loc'][1] + person_info['front_new_size'][1])
    #     # front_roi = train_source_image.crop(front_box)
    #     # front_roi.save('c:/temp/front_roi.jpg')
    #     #
    #     # front_box = (person_info['back_new_loc'][0], person_info['back_new_loc'][1],\
    #     #        person_info['back_new_loc'][0] + person_info['back_new_size'][0], person_info['back_new_loc'][1] + person_info['back_new_size'][1])
    #     # front_roi = train_source_image.crop(front_box)
    #     # front_roi.save('c:/temp/back_roi.jpg')
    #
    #     # 正面
    #     front_img_512 = SaveSingleSide(train_target_image, person_info['front_new_loc'], person_info['front_new_size'])
    #     # 反面
    #     back_img_512 = SaveSingleSide(train_target_image, person_info['back_new_loc'], person_info['back_new_size'])
    #
    #     front_img_512.save('c:/temp/front_roi.jpg')
    #     back_img_512.save('c:/temp/back_roi.jpg')
    #     # # 保存图片（灰度图)
    #     train_source_image.save('c:/temp/test_source.jpg',quality=95)
    #     train_target_image.save('c:/temp/test_target.jpg',quality=95)

    # ----------多线程 测试（文件生成）----------
    # thread_01 = Thread(target=ThreadCardMaker,args=(25,"t01",))
    # thread_01.start()
    #
    # thread_02 = Thread(target=ThreadCardMaker,args=(25,"t02",))
    # thread_02.start()
    #
    # thread_03 = Thread(target=ThreadCardMaker,args=(25,"t03",))
    # thread_03.start()
    #
    # thread_04 = Thread(target=ThreadCardMaker,args=(25,"t04",))
    # thread_04.start()

