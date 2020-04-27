# encoding:utf-8
"""
@Time    : 2020-02-24 14:46
@Author  : yshhuang@foxmail.com
@File    : midia.py
@Software: PyCharm
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data//1.jpg')
median = cv2.medianBlur(img, 5)
cv2.imwrite('median.png', median)