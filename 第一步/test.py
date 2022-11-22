# _*__coding:utf-8 _*__
# @Time :2021/7/5 9:55
# @Author :bay
# @File test.py
# @Software : PyCharm
import numpy as np

a = np.array([[1, 1, 2, 1, 5],
 [9, 9, 9, 9, 9],
 [9, 9, 9, 9, 9],
 [1, 1, 2, 1, 5],
 [1, 1, 2, 1, 5],
 [1, 1, 2, 1, 5],
 [1, 1, 2, 1, 5],
 [1, 1, 2, 1, 5]])
print(a)
b = np.array([[1,  2,  3,  1, 17],
 [1,  2,  3,  1, 17],
 [1,  2,  3,  1, 17],
 [1,  2,  3,  1, 17],
 [1,  2,  3,  1, 17]])
print(b)
c = a.tolist()
print(c)