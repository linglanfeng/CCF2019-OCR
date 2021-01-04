# #  民族单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/1018_v5_nation_only_46.csv'
# output_path = 'single_submit/1018_v5_nation_only_46.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
# df.columns = ['图片名字', '民族']
#
# df_src.loc[:,:] = 0
# df_src['民族'] = df['民族']
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()
#
# #  月 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/1019_month_only_93.csv'
# output_path = 'single_submit/1019_month_only_93.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '月']
#
# df_src.loc[:,:] = 0
# df_src['月'] = df['月'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()


#
#  # 日 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# # result_path = 'result/day_result_mix.csv'
# result_path = 'single_result/1019_day_only_47.csv'
# output_path = 'single_submit/1019_day_only_47.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '日']
#
# df_src.loc[:,:] = 0
# df_src['日'] = df['日'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()

#
#  # 年 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# # result_path = 'result/day_result_mix.csv'
# result_path = 'single_result/1019_year_only_31.csv'
# output_path = 'single_submit/1019_year_only_31.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '年']
#
# df_src.loc[:,:] = 0
# df_src['年'] = df['年'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()

# # sex单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/0907_sex_only_100.csv'
# output_path = 'single_submit/0907_sex_only_100.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
# df.columns = ['图片名字', '性别']
#
# df_src.loc[:,:] = 0
# df_src['性别'] = df['性别']
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()


 # ID 单列提交
import pandas as pd

src_path = 'single_submit/submit_example.csv'
result_path = 'single_result/1021_id_only_33_t1017.csv'
output_path = 'single_submit/1021_id_only_33_t1017.csv'

df_src = pd.read_csv(src_path, header=None)
df = pd.read_csv(result_path, header=None)
columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
df_src.columns = columns

df.columns = ['图片名字', '公民身份号码']

df_src.loc[:,:] = 0
df_src['公民身份号码'] = df['公民身份号码'].astype('str')
# df_src['月'] = df_src['月'].astype('int')
df_src['图片名字'] = df['图片名字']

df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
print()

#
#
#  # 有效期限 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/1020_v4_valid_only_29_t1017.csv'
# output_path = 'single_submit/1020_v4_valid_only_29_t1017.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '有效期限']
#
# df_src.loc[:,:] = 0
# df_src['有效期限'] = df['有效期限'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()

#
# # 名字 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/1013_name_only_400_t1017.csv'
# output_path = 'single_submit/1013_name_only_400_t1017.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '姓名']
#
# df_src.loc[:,:] = 0
# df_src['姓名'] = df['姓名'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()

#
#
# # 地址 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# # result_path = 'single_result/1006_address_only_180_decode_ctc_l2_correct_cut_all_top150.csv'
# # output_path = 'single_submit/1006_address_only_180_decode_ctc_l2_correct_cut_all_top150.csv'
#
# # result_path = 'single_result/1005_address_only_40.csv'
# # output_path = 'single_submit/1005_address_only_40.csv'
#
# # result_path = 'single_result/1006_address_only_180_decode_ctc_l2_correct_cut_all_top150_v2.csv'
# # output_path = 'single_submit/1006_address_only_180_decode_ctc_l2_correct_cut_all_top150_v2.csv'
#
# result_path = 'single_result/1006_address_only_180_1014test.csv'
# output_path = 'single_submit/1006_address_only_180_1014test.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '住址']
#
# df_src.loc[:,:] = 0
# df_src['住址'] = df['住址'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()


#
# # 发证机关 单列提交
# import pandas as pd
#
# src_path = 'single_submit/submit_example.csv'
# result_path = 'single_result/1008_office_only_33_l2_correct.csv'
# output_path = 'single_submit/1008_office_only_33_l2_correct.csv'
#
# df_src = pd.read_csv(src_path, header=None)
# df = pd.read_csv(result_path, header=None)
# columns = ['图片名字', '姓名', '民族', '性别', '年', '月', '日', '住址', '公民身份号码', '签发机关', '有效期限']
# df_src.columns = columns
#
# df.columns = ['图片名字', '签发机关']
#
# df_src.loc[:,:] = 0
# df_src['签发机关'] = df['签发机关'].astype('str')
# # df_src['月'] = df_src['月'].astype('int')
# df_src['图片名字'] = df['图片名字']
#
# df_src.to_csv(output_path, encoding='utf_8_sig', index=False, header=None)
# print()