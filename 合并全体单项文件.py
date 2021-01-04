# 地址 单列提交
import pandas as pd

src_path = 'single_submit/submit_example.csv'

# output_path = 'all_submit/1015_best_submit_all.csv' #99.511%  ->0.995407
# best_result_file = {'姓名':'1013_name_only_400.csv'
#                     ,'民族':'0905_nation_only_mix_ext1w.csv'
#                     ,'性别':'0907_sex_only_mix.csv'
#                     ,'年':'0908_year_only_mix_ext1w.csv'
#                     ,'月':'0906_month_only_mix.csv'
#                     ,'日':'0907_day_only_mix_ext1w.csv'
#                     ,'住址':'1006_address_only_180_1014test.csv'
#                     ,'公民身份号码':'1015_id_only_50.csv'
#                     ,'签发机关':'1008_office_only_33_l2_correct.csv'
#                     ,'有效期限':'1012_valid_only_200.csv'
#                     ,}

# output_path = 'all_submit/1020_best_submit_all.csv' #99.651%  -> 0.99660
# best_result_file = {'姓名':'1013_name_only_400_t1017.csv'
#                     ,'民族':'1018_v5_nation_only_46.csv'
#                     ,'性别':'0907_sex_only_mix.csv'
#                     ,'年':'1019_year_only_31.csv'
#                     ,'月':'1019_month_only_93.csv'
#                     ,'日':'1019_day_only_47.csv'
#                     ,'住址':'1006_address_only_180_1014test.csv'
#                     ,'公民身份号码':'1015_id_only_50.csv'
#                     ,'签发机关':'1008_office_only_33_l2_correct.csv'
#                     ,'有效期限':'1012_valid_only_200.csv'
#                     ,}

output_path = 'all_submit/1021_best_submit_all.csv' #99.687%  -> 0.996958
best_result_file = {'姓名':'1013_name_only_400_t1017.csv'
                    ,'民族':'1018_v5_nation_only_46.csv'
                    ,'性别':'0907_sex_only_mix.csv'
                    ,'年':'1019_year_only_31.csv'
                    ,'月':'1019_month_only_93.csv'
                    ,'日':'1019_day_only_47.csv'
                    ,'住址':'1006_address_only_180_1014test.csv'
                    ,'公民身份号码':'1015_id_only_50.csv'
                    ,'签发机关':'1008_office_only_33_l2_correct.csv'
                    ,'有效期限':'1020_v4_valid_only_29_t1017.csv'
                    ,}

df_src = pd.read_csv(src_path, header=None)
columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
df_src.columns = columns
df_src.loc[:, 1:] = 0
df_src.set_index(['图片名字'],inplace=True)

for idx,col in enumerate(columns[1:]):
    print(col)

    result_path = 'single_submit/{0}'.format(best_result_file[col])

    print(result_path)
    df = pd.read_csv(result_path, header=None, names=columns,dtype={'年':'str', '月':'str', '日':'str','公民身份号码':'str'})
    df = df.loc[~df.图片名字.isna()].copy()

    df.set_index(['图片名字'], inplace=True)
    df.fillna('0', inplace=True)
    # df.columns = columns
    df_src[col] = df[col].astype('str')
    # df_src['图片名字'] = df['图片名字']



df_src.fillna('0',inplace=True)
df_src.reset_index(inplace=True)
df_src.loc[df_src.图片名字 == '0','图片名字'] = ''
df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
print(22)