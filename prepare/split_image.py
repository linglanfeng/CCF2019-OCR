import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
import PIL.ImageOps as ImageOps
from PIL import Image
import os

def rotate_bound(image, angle):
    """
    图像旋转
    :param image:
    :param angle:
    :return:
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

class SplitImage(object):

    # 1017 识别时候坐标已经修改，图片特别接近就存在问题
    # merge_blank_width = 25
    # merge_blank_height = 15

    merge_blank_width = 0
    merge_blank_height = 0

    # 保存路径
    # root_path = 'temp/card_maker_temp'
    root_path = 'temp/split_1017'

    # 切分文件位置（子目录）
    nationality_path = root_path + '/nationality'
    sex_path = root_path + '/sex'
    name_path = root_path + '/name'
    year_path = root_path + '/year'
    month_path = root_path + '/month'
    day_path = root_path + '/day'
    addr_path = root_path + '/addr'
    id_path = root_path + '/id'
    office_path = root_path + '/office'
    valid_path = root_path + '/valid'
    test_path = root_path + '/test_unet'

    nationality_dict = {}
    sex_dict = {}

    table = []
    def get_threshold_table(self):
        threshold = 255 - 125
        temp_table = []
        for i in range(256):
            if i < threshold:
                temp_table.append(0)
            else:
                temp_table.append(1)
        return temp_table

    def get_max_loc(self, card_location):
        max_index = 0
        max_value = 0
        for idx, loc in enumerate(card_location):
            if loc[1] - loc[0] > max_value:
                max_value = loc[1] - loc[0]
                max_index = idx

        return card_location[max_index]

    def get_location(self, p_image, p_axis=1, p_window=9, p_percentile=50):

        temp_front_img = cv2.equalizeHist(p_image)
        temp_front_img = ImageOps.invert(Image.fromarray(temp_front_img)).point(self.table, '1')

        axis_sum = np.sum(temp_front_img, axis=p_axis)
        axis_sum = axis_sum * (axis_sum > p_percentile)
        axis_sum[0] = 0

        axis_sum[axis_sum > 0] = 1

        axis_sum = pd.DataFrame(data=axis_sum).rolling(p_window, center=True, min_periods=3).max()[0]

        image_flag = np.diff(np.concatenate(([0], axis_sum)))

        # 奇数
        if np.argwhere(image_flag != 0).shape[0] % 2 == 1:
            image_flag[len(image_flag) - 1] = -1

        image_loc = np.argwhere(image_flag != 0).reshape(-1, 2)

        del temp_front_img

        return self.get_max_loc(image_loc)

    def __init__(self):
        # 阈值表
        self.table = self.get_threshold_table()

        # ------------初始化文件夹-------------
        # 创建姓名文件夹
        if not os.path.exists(self.name_path):
            os.makedirs(self.name_path)

        # 创建性别文件夹
        if not os.path.exists(self.sex_path):
            os.makedirs(self.sex_path)

        # 创建民族文件夹
        if not os.path.exists(self.nationality_path):
            os.makedirs(self.nationality_path)

        # 创建生日文件夹
        if not os.path.exists(self.year_path):
            os.makedirs(self.year_path)
        if not os.path.exists(self.month_path):
            os.makedirs(self.month_path)
        if not os.path.exists(self.day_path):
            os.makedirs(self.day_path)

        # 创建住址文件夹
        if not os.path.exists(self.addr_path):
            os.makedirs(self.addr_path)

        # 创建身份证号码文件夹
        if not os.path.exists(self.id_path):
            os.makedirs(self.id_path)

        # 创建发证机关文件夹
        if not os.path.exists(self.office_path):
            os.makedirs(self.office_path)

        # 创建有效期文件夹
        if not os.path.exists(self.valid_path):
            os.makedirs(self.valid_path)

        # 创建TEST文件夹
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        #  创建 保存图片的文件夹（110年）
        for idx in range(1910, 2020):
            if not os.path.exists(self.year_path + "/" + str(idx)):
                os.makedirs(self.year_path + "/" + str(idx))
        #  创建 保存图片的文件夹（12月）
        for idx in range(1, 13):
            if not os.path.exists(self.month_path + "/" + str(idx)):
                os.makedirs(self.month_path + "/" + str(idx))
        #  创建 保存图片的文件夹（31日）
        for idx in range(1, 32):
            if not os.path.exists(self.day_path + "/" + str(idx)):
                os.makedirs(self.day_path + "/" + str(idx))


        # 创建 保存图片的文件夹（56民族）
        # {'土家': '0', '塔吉克': '1', '汉': '2', '佤': '3', '撒拉': '4', '白': '5', '高山': '6', '傣': '7', '纳西': '8', '鄂温克': '9', '畲': '10', '回': '11', '藏': '12', '傈僳': '13', '达斡尔': '14', '锡伯': '15', '塔塔尔': '16', '保安': '17', '独龙': '18', '珞巴': '19', '布依': '20', '景颇': '21', '赫哲': '22', '基诺': '23', '侗': '24', '乌孜别克': '25', '布朗': '26', '门巴': '27', '崩龙': '28', '土': '29', '朝鲜': '30', '蒙古': '31', '东乡': '32', '哈萨克': '33', '黎': '34', '水': '35', '仫佬': '36', '京': '37', '怒': '38', '阿昌': '39', '哈尼': '40', '拉祜': '41', '羌': '42', '柯尔克孜': '43', '满': '44', '裕固': '45', '毛难': '46', '瑶': '47', '维吾尔': '48', '壮': '49', '鄂伦春': '50', '俄罗斯': '51', '仡佬': '52', '彝': '53', '苗': '54', '普米': '55'}
        for idx, dir_name in enumerate(['土家', '塔吉克', '汉', '佤', '撒拉', '白', '高山', '傣', '纳西', '鄂温克', '畲', '回',
                                         '藏', '傈僳', '达斡尔', '锡伯', '塔塔尔', '保安', '独龙', '珞巴', '布依', '景颇', '赫哲',
                                         '基诺', '侗', '乌孜别克', '布朗', '门巴', '崩龙', '土', '朝鲜', '蒙古', '东乡', '哈萨克',
                                         '黎', '水', '仫佬', '京', '怒', '阿昌', '哈尼', '拉祜', '羌', '柯尔克孜', '满', '裕固',
                                         '毛难', '瑶', '维吾尔', '壮', '鄂伦春', '俄罗斯', '仡佬', '彝', '苗', '普米']):
            self.nationality_dict[dir_name] = str(idx)
            if not os.path.exists(self.nationality_path + "/" + self.nationality_dict[dir_name]):
                os.makedirs(self.nationality_path + "/" + self.nationality_dict[dir_name])

        # 创建 保存图片的文件夹（性别）
        for idx, dir_name in enumerate(['男','女']):
            self.sex_dict[dir_name] = str(idx)
            if not os.path.exists(self.sex_path + "/" + self.sex_dict[dir_name]):
                os.makedirs(self.sex_path + "/" + self.sex_dict[dir_name])

    def split_image(self, img, yolo_row, auto_scale = 0):

        max_size = img.shape[1]

        # 根据坐标切分图片
        front_img = img[max(yolo_row['front_y1'] - self.merge_blank_height,0):min(yolo_row['front_y2'] + self.merge_blank_height, max_size),
                        max(yolo_row['front_x1'] - self.merge_blank_width,0):min(yolo_row['front_x2'] + self.merge_blank_width, max_size)]

        back_img = img[ max(yolo_row['back_y1'] - self.merge_blank_height,0):min(yolo_row['back_y2'] + self.merge_blank_height, max_size),
                        max(yolo_row['back_x1'] - self.merge_blank_width,0):min(yolo_row['back_x2'] + self.merge_blank_width, max_size)]

        # 修正图片（旋转）
        if yolo_row['front_rotate_180'] == 0:
            front_img = ndimage.rotate(front_img, yolo_row['front_rotate_angle'], cval=255)
            front_img = rotate_bound(front_img, 180)
        else:
            front_img = ndimage.rotate(front_img, yolo_row['front_rotate_angle'], cval=255)

        if yolo_row['back_rotate_180'] == 0:
            back_img = ndimage.rotate(back_img, yolo_row['back_rotate_angle'], cval=255)
            back_img = rotate_bound(back_img, 180)
        else:
            back_img = ndimage.rotate(back_img, yolo_row['back_rotate_angle'], cval=255)

        front_y_loc = self.get_location(front_img)
        back_y_loc = self.get_location(back_img)

        front_x_loc = self.get_location(front_img[front_y_loc[0]:front_y_loc[1], :],p_axis = 0, p_percentile=20)
        back_x_loc = self.get_location(back_img[back_y_loc[0]:back_y_loc[1], :], p_axis=0, p_percentile=20)

        padding = 5  #9//2
        front_left_top_loc = [front_x_loc[0] + padding, front_y_loc[0] + padding]
        front_right_bottom_loc = [front_x_loc[1] + padding, front_y_loc[1] + padding]

        back_left_top_loc = [back_x_loc[0] + padding, back_y_loc[0] + padding]
        back_right_bottom_loc = [back_x_loc[1] + padding, back_y_loc[1] + padding]

        front_card_image_width = front_right_bottom_loc[0] - front_left_top_loc[0]
        front_card_image_height = front_right_bottom_loc[1] - front_left_top_loc[1]

        back_card_image_width = back_right_bottom_loc[0] - back_left_top_loc[0]
        back_card_image_height = back_right_bottom_loc[1] - back_left_top_loc[1]

        main_front_img = front_img[front_left_top_loc[1]:front_right_bottom_loc[1]
                                , front_left_top_loc[0]:front_right_bottom_loc[0]]

        main_back_img = back_img[back_left_top_loc[1]:back_right_bottom_loc[1]
                                , back_left_top_loc[0]:back_right_bottom_loc[0]]

        # print(back_card_image_width,back_card_image_height)
        # 图片缩放(等比例缩放)
        front_scale_rate = 1.0
        back_scale_rate = 1.0
        if auto_scale == 1:
            front_scale_rate = round(max(452 / front_card_image_width, 290 / front_card_image_height),2)
            back_scale_rate = round(max(465 / back_card_image_width, 298 / back_card_image_height),2)
        # front_scale_rate = round(452 / front_card_image_width,2)
        # back_scale_rate = round(465 / back_card_image_width,2)

        if front_scale_rate != 1.0 :
            print('front_scale_rate =', front_scale_rate)
            main_front_img = cv2.resize(main_front_img, (int(main_front_img.shape[1] * front_scale_rate), int(main_front_img.shape[0] * front_scale_rate)), interpolation=cv2.INTER_CUBIC)

        if back_scale_rate != 1.0 :
            print('back_scale_rate =' ,back_scale_rate)
            main_back_img = cv2.resize(main_back_img, (int(main_front_img.shape[1] * back_scale_rate), int(main_front_img.shape[0] * back_scale_rate)), interpolation=cv2.INTER_CUBIC)

        # print('scale:', front_scale_rate, back_scale_rate)

        # 姓名
        name_img = main_front_img[ 38 : 38 + 40
                                  ,68 : 68 + 110]
        # cv2.imwrite(self.name_path + '/{0}_0.jpg'.format(yolo_row['file']), name_img)

        # 性别
        sex_img = main_front_img[70 : 70 + 40
                                ,68 : 68 + 60]
        # cv2.imwrite(self.sex_path + '/{0}/{1}_1.jpg'.format(self.sex_dict[info['sex'].values[0]],yolo_row['file']), sex_img)

        # 民族
        nationality_img = main_front_img[70 : 70 + 40
                                        ,170 : 170 + 120]
        # cv2.imwrite(self.nationality_path +'/{0}/{1}_2.jpg'.format(self.nationality_dict[info['nationality'].values[0]], yolo_row['file']), nationality_img)

        # 年
        year_img = main_front_img[102 : 102 + 40
                                  ,68 :68 + 70]
        # cv2.imwrite(self.year_path + '/{0}/{1}_3.jpg'.format(info['birthday_y'].values[0],yolo_row['file']), year_img)

        # 月
        month_img = main_front_img[102 : 102 + 40
                                  ,132 : 132 + 52]
        # cv2.imwrite(self.month_path + '/{0}/{1}_4.jpg'.format(info['birthday_m'].values[0],yolo_row['file']), month_img)

        # 日
        day_img = main_front_img[102 : 102 + 40
                                ,182 : 182 + 52]
        # cv2.imwrite(self.day_path + '/{0}/{1}_5.jpg'.format(info['birthday_d'].values[0],yolo_row['file']), day_img)

        # 住址
        addr_img = main_front_img[130: 130 + 105
                                  ,68: 68 + 250]
        # addr_img = main_front_img[135: 135 + 100
        #                           ,68: 68 + 250] #精细调整（效果不好）
        # cv2.imwrite(self.addr_path + '/{0}_6.jpg'.format(yolo_row['file']), addr_img)

        # 身份证号码
        id_img = main_front_img[ 224: 224 + 40
                                ,110: 110 + 300]
        # cv2.imwrite(self.id_path + '/{0}_7.jpg'.format(yolo_row['file']), id_img)

        # 发证机关
        # office_img = main_back_img[192: 192 + 52
        #                            ,165: 165 + 240]
        # office_img = main_back_img[192: 192 + 45
        #                            ,165: 165 + 240]
        office_img = main_back_img[195: 195 + 45
                                   ,165: 165 + 240]  # 1017修改
        # cv2.imwrite(self.office_path + '/{0}_8.jpg'.format(yolo_row['file']), office_img)

        # 有效期限
        # valid_img = main_back_img[ 232: 232 + 52
        #                            ,165: 165 + 240]
        valid_img = main_back_img[ 230: 230 + 35
                                   ,165: 165 + 240] #精细调整
        # cv2.imwrite(self.valid_path + '/{0}_9.jpg'.format(yolo_row['file']), valid_img)

        return {'name':name_img
                ,'sex':sex_img
                ,'nationality':nationality_img
                ,'year':year_img
                ,'month' : month_img
                ,'day' : day_img
                ,'addr' : addr_img
                ,'id' : id_img
                ,'office': office_img
                ,'valid': valid_img
                }

#
# if __name__ == '__main__':
#     split = SplitImage()
#
#     # yolo 目标识别后的数据
#     yolo_pd = pd.read_csv('output_imgs/yolo_rect_train.csv')
#     image_path = 'E:/projects/python/DF_idcard/data/train_data/{0}.jpg'
#
#     # 训练数据
#     data_pd = pd.read_csv('E:/projects/python/DF_idcard/data/Train_Labels.csv',header=None)
#     data_pd.columns = ['file', 'name', 'nationality', 'sex', 'birthday_y', 'birthday_m', 'birthday_d', 'addr', 'id_code', 'office', 'valid']
#
#     for index, yolo_row in yolo_pd.iterrows():
#
#         file_name = image_path.format(yolo_row['file'])
#         img = cv2.imread(file_name, 0)
#
#         image_info = data_pd.loc[data_pd.file == yolo_row['file']]
#
#         sub_images_dict = split.split_image(img, yolo_row)
#
#         # --------------图片保存------------
#         # 姓名
#         cv2.imwrite(split.name_path + '/{0}_0.jpg'.format(yolo_row['file']), sub_images_dict['name'])
#         # 性别
#         cv2.imwrite(split.sex_path + '/{0}/{1}_1.jpg'.format(split.sex_dict[image_info['sex'].values[0]], yolo_row['file']), sub_images_dict['sex'])
#         # 民族
#         cv2.imwrite(split.nationality_path + '/{0}/{1}_2.jpg'.format(split.nationality_dict[image_info['nationality'].values[0]],yolo_row['file']), sub_images_dict['nationality'])
#         # 生日（年）
#         cv2.imwrite(split.year_path + '/{0}/{1}_3.jpg'.format(image_info['birthday_y'].values[0], yolo_row['file']), sub_images_dict['year'])
#         # 生日（月）
#         cv2.imwrite(split.month_path + '/{0}/{1}_4.jpg'.format(image_info['birthday_m'].values[0], yolo_row['file']),sub_images_dict['month'])
#         # 生日（日）
#         cv2.imwrite(split.day_path + '/{0}/{1}_5.jpg'.format(image_info['birthday_d'].values[0], yolo_row['file']), sub_images_dict['day'])
#         # 住址
#         cv2.imwrite(split.addr_path + '/{0}_6.jpg'.format(yolo_row['file']), sub_images_dict['addr'])
#         # 身份证号码
#         cv2.imwrite(split.id_path + '/{0}_7.jpg'.format(yolo_row['file']), sub_images_dict['id'])
#         # 发证机关
#         cv2.imwrite(split.office_path + '/{0}_8.jpg'.format(yolo_row['file']),sub_images_dict['office'])
#         # 有效日期
#         cv2.imwrite(split.valid_path + '/{0}_9.jpg'.format(yolo_row['file']), sub_images_dict['valid'])
#
#         print(index, yolo_row['file'])
