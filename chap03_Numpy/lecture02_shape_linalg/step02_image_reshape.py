# -*- coding: utf-8 -*-
"""
step02_image_reshape.py
"""

import matplotlib.pyplot as plt  
from sklearn.datasets import load_digits 


# 1. image shape & reshape

# 1) dataset load 
digits = load_digits() # 머신러닝 모델에서 사용되는 데이터셋 
dir(digits) # data, target..

'''
입력변수(X) : 숫자(0~9) 필기체의 흑백 이미지 
출력변수(y) : 10진 정수
'''
X = digits.data # 입력변수(X) 추출 
y = digits.target # 출력변수(y) 추출 
X.shape # (1797, 64) : (size, pixel정보)

X[0].shape # (64,) # 64개의 원소 
# 0 ~ 1796 개의 세트가 있음 ★★★
# 2) image reshape 
first_img = X[11].reshape(8,8) # 모양변경 : 2d(h,w) 
first_img.shape # (8, 8)

# 3) image show 
plt.imshow(X=first_img, cmap='gray')
plt.show()

# 첫번째 이미지 정답 
y[0] # 0

# 마지막 이미지 
last_img = X[-1].reshape(8,8) # 모양변경 : 2d(h,w)
# last_img = X[1796].reshape(8,8)과 동일
plt.imshow(X=last_img, cmap='gray')
plt.show()

y[-1] # 8


# 2. image file read & show
import matplotlib.image as img # 이미지 읽기 

# image file path 
path = r"C:/ITWILL/4_Python_ML/Python_ML/chap03_Numpy" # 이미지 경로 

# 1) image 읽기 
# img.imread(경로)
img_arr = img.imread(path + "/data/test1.jpg")
type(img_arr) # numpy.ndarray


img_arr.shape #  (360, 540, 3) : (h, w, color)

# 2) image 출력 
plt.imshow(img_arr)
plt.show()

