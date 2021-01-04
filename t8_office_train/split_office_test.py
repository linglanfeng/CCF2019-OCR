import pandas as pd
import numpy as np
import cv2
import PIL.ImageOps as ImageOps
from PIL import Image
import os
import math

class SplitAddress(object):

    # 滤波核大小
    kernel = np.ones((9, 9), np.uint8)
    text_height_panding = 2
    text_width_panding = 25*5
    text_height = 25

    def get_threshold_table(self):
        threshold = 255 - 40
    #     threshold = int(p_threshold*1.1)
        temp_table = []
        for i in range(256):
            if i < threshold:
                temp_table.append(0)
            else:
                temp_table.append(1)
        return temp_table

    def get_max_loc(self, card_location):

        if len(card_location) == 0:
            return []

        max_index = 0
        max_value = 0
        for idx, loc in enumerate(card_location):
            if loc[1] - loc[0] > max_value:
                max_value = loc[1] - loc[0]
                max_index = idx

        return card_location[max_index]

    def get_location(self, p_image, p_axis=1, p_window=9, p_percentile=50):

        # 腐蚀
        erosion = cv2.erode(p_image, self.kernel)

        temp_front_img = cv2.equalizeHist(erosion) #两次腐蚀后不需要再均衡化
        temp_front_img = ImageOps.invert(Image.fromarray(temp_front_img)).point(self.get_threshold_table(), '1')

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

    def get_text_line(self,p_image,y_text_loc,test_file_name, max_line = 4):

        # 设置最大行数(最少1行) 每行行高 25
        text_line = int(max(min(round((y_text_loc[1] - y_text_loc[0]) / self.text_height, 0),max_line),1))

        # 计算单行高度
        text_split = int((y_text_loc[1] - y_text_loc[0]) / text_line)
        if text_split < self.text_height - 3:
            text_split = self.text_height - 3

        # print(p_image.shape)

        text_xy_loc = []
        for i in range(text_line):

            # 行数据
            top_left_y = max(y_text_loc[0] + text_split * i - self.text_height_panding - i*3, 0)
            bottom_right_y = min(y_text_loc[0] + text_split * (i+1) + self.text_height_panding  - i*3, p_image.shape[0])

            # left_x = min(x_text_loc[0],10)
            top_left_x = 0

            #第一行不截取
            if i == 0:
                bottom_right_x = p_image.shape[1]
            else:
                # 获得文字长度
                x_text_loc = self.get_location(p_image[top_left_y: bottom_right_y, :], p_axis=0, p_window=40, p_percentile=7)

                # 长度为零
                if len(x_text_loc) == 0:
                    print(test_file_name, '------------------> x_text_loc')
                    continue
                bottom_right_x = min(x_text_loc[1] + self.text_width_panding, p_image.shape[1])

            text_xy_loc.append([top_left_x,top_left_y, bottom_right_x, bottom_right_y])

            # print(line_loc,'$',x_text_loc)

        return text_xy_loc

    def split_image(self, img, test_file_name, auto_scale = 0):

        y_text_loc = self.get_location(img, p_window=9, p_percentile=5)
        text_line_loc  = self.get_text_line(img, y_text_loc,test_file_name, 4)

        return text_line_loc


if __name__ == '__main__':
    split = SplitAddress()


    image_path = 'D:/projects/data/ccf2019_ocr/office_l2/train_office_part'

    # # 训练数据
    # data_pd = pd.read_csv('D:/projects/data/ccf2019_ocr//Train_Labels.csv',header=None)
    # data_pd.columns = ['file', 'name', 'nationality', 'sex', 'birthday_y', 'birthday_m', 'birthday_d', 'addr', 'id_code', 'office', 'valid']
    # label_file = open('D:/projects/data/ccf2019_ocr/address/train_txt/address_lable_by_df_train.txt', 'w')

    error_cnt = 0
    for root, dirs, files in os.walk(image_path):
        for file in files:

            # file = 'E:/projects/python/DF_idcard_all/temp/addr/141dad7d8169416c962c4ccfef5c9764_6.jpg'
            file_name = file[:-4]
            # if file_name.startswith('s_'):
            #     continue

            if not file[-6:] == "_8.jpg":
                continue

            file_name = file[:-6]

            src_file = os.path.join(root, file)
            # des_file = os.path.join(image_path, file)

            img = cv2.imread(src_file, 0)
            # des_img = cv2.imread(des_file, 0)

            sub_images_dict = split.split_image(img, file_name)

            for idx,text_line in enumerate(sub_images_dict):
                sub_file_name = '{0}_{1}.jpg'.format(file_name,idx)
                print(sub_file_name)
                cv2.imwrite('D:/projects/data/ccf2019_ocr/office_l2/test_img/{0}'.format(sub_file_name), img[text_line[1]:text_line[3], text_line[0]:text_line[2]])


    # file_name = image_path.format(image_path)
    # img = cv2.imread(file_name, 0)

    # image_info = data_pd.loc[data_pd.file == yolo_row['file']]

    # sub_images_dict = split.split_image(img, "")

    # --------------图片保存------------
    # # 姓名
    # cv2.imwrite(split.name_path + '/{0}_0.jpg'.format(yolo_row['file']), sub_images_dict['name'])
    # # 性别
    # cv2.imwrite(split.sex_path + '/{0}/{1}_1.jpg'.format(split.sex_dict[image_info['sex'].values[0]], yolo_row['file']), sub_images_dict['sex'])
    # # 民族
    # cv2.imwrite(split.nationality_path + '/{0}/{1}_2.jpg'.format(split.nationality_dict[image_info['nationality'].values[0]],yolo_row['file']), sub_images_dict['nationality'])
    # # 生日（年）
    # cv2.imwrite(split.year_path + '/{0}/{1}_3.jpg'.format(image_info['birthday_y'].values[0], yolo_row['file']), sub_images_dict['year'])
    # # 生日（月）
    # cv2.imwrite(split.month_path + '/{0}/{1}_4.jpg'.format(image_info['birthday_m'].values[0], yolo_row['file']),sub_images_dict['month'])
    # # 生日（日）
    # cv2.imwrite(split.day_path + '/{0}/{1}_5.jpg'.format(image_info['birthday_d'].values[0], yolo_row['file']), sub_images_dict['day'])
    # # 住址
    # cv2.imwrite(split.addr_path + '/{0}_6.jpg'.format(yolo_row['file']), sub_images_dict['addr'])
    # # 身份证号码
    # cv2.imwrite(split.id_path + '/{0}_7.jpg'.format(yolo_row['file']), sub_images_dict['id'])
    # # 发证机关
    # cv2.imwrite(split.office_path + '/{0}_8.jpg'.format(yolo_row['file']),sub_images_dict['office'])
    # # 有效日期
    # cv2.imwrite(split.valid_path + '/{0}_9.jpg'.format(yolo_row['file']), sub_images_dict['valid'])

    # print(sub_images_dict)
    #
    # print('aid')




