import pandas as pd
import cv2
from split_image import SplitImage
from predict_main import Unet
import os
'''
Ver 0904 修正位置
'''
def create_train():

    output_path = 'output_imgs'  # 保存中间文件的路径


    split = SplitImage()

    # yolo 目标识别后的数据
    yolo_pd = pd.read_csv('output_imgs/yolo_rect_train_1017.csv')
    image_path = 'E:/projects/python/DF_idcard/data/train_data/{0}.jpg'
    # 训练数据
    data_pd = pd.read_csv('E:/projects/python/DF_idcard/data/Train_Labels.csv', header=None)
    data_pd.columns = ['file', 'name', 'nationality', 'sex', 'birthday_y', 'birthday_m', 'birthday_d', 'addr',
                       'id_code', 'office', 'valid']

    # yolo_pd = pd.read_csv('output_imgs/yolo_rect_card_maker.csv')
    # image_path = 'C:/temp/ALL_IN_ONE_TRAIN/with_stamp/{0}.jpg'
    # # 训练数据
    # data_pd = pd.read_csv('C:/temp/ALL_IN_ONE_TRAIN/data_all.csv')
    # data_pd['file_name'] = data_pd.id_code + '_' + data_pd.uuid
    # data_pd = data_pd[['file_name','name','nationality','sex','birthday_y','birthday_m','birthday_d','addr','id_code','office','valid']]
    # data_pd.columns = ['file', 'name', 'nationality', 'sex', 'birthday_y', 'birthday_m', 'birthday_d', 'addr', 'id_code', 'office', 'valid']

    # split.root_path = 'temp/card_maker_temp'

    unet = Unet()  # 加载Unet模型

    file_list = []
    name_list = []
    addr_list = []
    id_list = []
    office_list = []
    valid_list = []

    # with open(split.root_path + '/valid_lable_by_df_train.txt', 'w') as f:
    #     f.write("\n")

    for index, yolo_row in yolo_pd.iterrows():
        file_name = image_path.format(yolo_row['file'])

        # --------------------原图
        img_leave = cv2.imread(file_name, 0)
        # 原图用s，滤波后用u
        # pre_name = 's_'
        pre_name = '' # 1012版本删除，不使用unet训练了

        # # --------------------unet对原图去水印
        # img = unet.predict(file_name)
        # cv2.imwrite(output_path + '/unet_img.jpg', img)
        # img_leave = cv2.imread(output_path + '/unet_img.jpg', 0)
        # # 原图用s，滤波后用"空白"
        # pre_name = ''

        image_info = data_pd.loc[data_pd.file == yolo_row['file']]

        # 传入图片 最好是 Unet 处理完毕的图片
        sub_images_dict = split.split_image(img_leave, yolo_row)

        # --------------保存列表-----------
        file_list.append(image_info['file'])
        name_list.append(image_info['name'])
        addr_list.append(image_info['addr'])
        id_list.append(image_info['id_code'])
        office_list.append(image_info['office'])
        valid_list.append(image_info['valid'])

        # with open(split.root_path+'/train_data_list.csv', 'a+') as f:
        #     f.write("{0},{1},{2},{3},{4},{5}\n".format(pre_name+image_info['file'].values[0],image_info['name'].values[0],image_info['addr'].values[0]
        #                                                ,image_info['id_code'].values[0],image_info['office'].values[0],image_info['valid'].values[0]))

        # # 注意 要修改 encoding='utf_8_sig',
        # with open(split.root_path+'/valid_lable_by_df_train.txt', 'a+') as f:
        #     f.write("{0} {1}\n".format(pre_name+image_info['file'].values[0]+"_9",image_info['valid'].values[0]))

        # # 注意 要修改 encoding='utf_8_sig',
        # with open(split.root_path+'/id_lable_by_df_train.txt', 'a+') as f:
        #     f.write("{0} {1}\n".format(pre_name+image_info['file'].values[0]+"_7",image_info['id_code'].values[0]))

        # --------------图片保存------------
        # # 姓名
        # cv2.imwrite(split.name_path + '/{0}{1}_0.jpg'.format(pre_name, yolo_row['file']), sub_images_dict['name'])
        # # 性别
        # cv2.imwrite(split.sex_path + '/{0}/{1}{2}_1.jpg'.format(split.sex_dict[image_info['sex'].values[0]], pre_name, yolo_row['file']),sub_images_dict['sex'])
        # # 民族
        # cv2.imwrite(split.nationality_path + '/{0}/{1}{2}_2.jpg'.format(split.nationality_dict[image_info['nationality'].values[0]], pre_name, yolo_row['file']),sub_images_dict['nationality'])
        # # 生日（年）
        # cv2.imwrite(split.year_path + '/{0}/{1}{2}_3.jpg'.format(image_info['birthday_y'].values[0], pre_name, yolo_row['file']), sub_images_dict['year'])
        # # 生日（月）
        # cv2.imwrite(split.month_path + '/{0}/{1}{2}_4.jpg'.format(image_info['birthday_m'].values[0], pre_name, yolo_row['file']), sub_images_dict['month'])
        # # 生日（日）
        # cv2.imwrite(split.day_path + '/{0}/{1}{2}_5.jpg'.format(image_info['birthday_d'].values[0], pre_name, yolo_row['file']), sub_images_dict['day'])
        # # 住址
        # cv2.imwrite(split.addr_path + '/{0}{1}_6.jpg'.format(pre_name, yolo_row['file']), sub_images_dict['addr'])
        # 身份证号码
        cv2.imwrite(split.id_path + '/{0}{1}_7.jpg'.format(pre_name, yolo_row['file']), sub_images_dict['id'])
        # # 发证机关
        # cv2.imwrite(split.office_path + '/{0}{1}_8.jpg'.format(pre_name, yolo_row['file']), sub_images_dict['office'])
        # # 有效日期
        # cv2.imwrite(split.valid_path + '/{0}{1}_9.jpg'.format(pre_name,yolo_row['file']), sub_images_dict['valid'])

        print(index, yolo_row['file'])

def create_test():

    output_path = 'output_imgs'  # 保存中间文件的路径

    split = SplitImage()

    # yolo 目标识别后的数据
    # yolo_pd = pd.read_csv('output_imgs/yolo_rect_card_maker_1009.csv')
    # yolo_pd = pd.read_csv('output_imgs/yolo_rect_test_1012.csv')
    # yolo_pd = pd.read_csv('output_imgs/yolo_rect_test_1014.csv')
    yolo_pd = pd.read_csv('output_imgs/yolo_rect_test_1017.csv')
    image_path = 'E:/projects/python/DF_idcard/data/Test/{0}.jpg'

    unet = Unet()  # 加载Unet模型

    for index, yolo_row in yolo_pd.iterrows():
        file_name = image_path.format(yolo_row['file'])

        # # --------------------原图（保存在不同的文件夹）
        # img_leave = cv2.imread(file_name, 0)

        # # --------------------unet对原图去水印（对地址定位有帮助）需要修改一下保存路径
        img = unet.predict(file_name)
        # 先刪除老图片
        # os.remove(output_path + '/unet_img.jpg')
        # 保存新图片
        cv2.imwrite(output_path + '/unet_img.jpg', img)
        img_leave = cv2.imread(output_path + '/unet_img.jpg', 0)

        # 传入图片 最好是 Unet 处理完毕的图片
        sub_images_dict = split.split_image(img_leave, yolo_row)

        # --------------图片保存------------
        # 姓名
        cv2.imwrite(split.test_path + '/{0}_0.jpg'.format(yolo_row['file']), sub_images_dict['name'])
        # 性别
        cv2.imwrite(split.test_path + '/{0}_1.jpg'.format(yolo_row['file']), sub_images_dict['sex'])
        # 民族
        cv2.imwrite(split.test_path + '/{0}_2.jpg'.format(yolo_row['file']), sub_images_dict['nationality'])
        # 生日（年）
        cv2.imwrite(split.test_path + '/{0}_3.jpg'.format(yolo_row['file']), sub_images_dict['year'])
        # 生日（月）
        cv2.imwrite(split.test_path + '/{0}_4.jpg'.format(yolo_row['file']), sub_images_dict['month'])
        # 生日（日）
        cv2.imwrite(split.test_path + '/{0}_5.jpg'.format(yolo_row['file']), sub_images_dict['day'])
        # 住址
        cv2.imwrite(split.test_path + '/{0}_6.jpg'.format(yolo_row['file']), sub_images_dict['addr'])
        # 身份证号码
        cv2.imwrite(split.test_path + '/{0}_7.jpg'.format(yolo_row['file']), sub_images_dict['id'])
        # 发证机关
        cv2.imwrite(split.test_path + '/{0}_8.jpg'.format(yolo_row['file']), sub_images_dict['office'])
        # 有效日期
        cv2.imwrite(split.test_path + '/{0}_9.jpg'.format(yolo_row['file']), sub_images_dict['valid'])

        print(index, yolo_row['file'])

        del img_leave

if __name__ == '__main__':
    # create_train()
    create_test()