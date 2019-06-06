from pandas import read_csv, set_option

filename = '/Users/dongsheng/Documents/me/resource/MachineLearning-master/chapter05/fpos_sales.csv'

names = ['store_code','clerk_no','quantity','retail_amount','coupon_amount','actual_amount']
data = read_csv(filename, names=names)
data['store_code'].astype(str)
data['clerk_no'].astype(str)
set_option('display.width',100)
#精确到小数点几位
set_option('precision',3)

print(data.dtypes)
#描述性统计
# print(data.describe())
#计算数据相关性
print(data.corr(method='pearson'))
# 数据的分布分析
# print(data.skew())