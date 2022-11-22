# _*__coding:utf-8 _*__
# @Time :2021/7/19 11:30
# @Author :bay
# @File time_log_read.py
# @Software : PyCharm
import pandas as pd
import numpy as np
# df = pd.read_csv('./time_log.csv', header=None)
# data = df.iloc[:, 1:]
# print(data)
# print(data.describe())
# print(data.unique())
import re

with open('time_log.txt', 'r') as f:
    datas = f.readlines()
    content_datas = []
    for data in datas:
        new_pa = re.findall(re.compile('(.*? :)(.*)'), data)
        new_data = new_pa[0][1]
        douhao_data = re.findall(re.compile('(.*?)  '), new_data)
        content_data = []
        for d in douhao_data:
            d = float(d)
            content_data.append(d)
        content_datas.append(content_data)
    print(content_datas)
    rst = [x for i in content_datas for x in i]
    print("_______________________________________________")
    print("rst", rst)
    print("________________________________________________")
    print(len(set(rst)), set(rst))
