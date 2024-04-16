# -*- coding: utf-8 -*-
"""
step03_image_resize.py

reshape : 모양변경(크기 변경 불가)
resize : 크기변경 -> 이미지 규격화(모양/크기)
"""

import numpy as np 
import matplotlib.pyplot as plt 

from PIL import Image # PIL(Python Image Lib) : open(), resize()


### 1. image resize
path = r"C:/ITWILL/4_Python_ML/Python_ML/chap03_Numpy" # 이미지 경로 

# 1) image read 
img = Image.open(path + "/data/test1.jpg") 
type(img) # PIL.JpegImagePlugin.JpegImageFile

np.shape(img) # 원본이미지 모양 : (360, 540, 3) 

# 2) image resize 
img_re = img.resize( (150, 120) )  # (가로, 세로)
type(img_re) # >> PIL.Image.Image
np.shape(img_re) # (120, 150, 3) 
# 배열의 차원을 알기 위해 np.shape(이미지객체)
img_ar = np.array(img_re)
print(img_ar)
img_ar.shape # (120, 150, 3) 동일

plt.imshow(img_re)
plt.show()


### 2. 폴더 전체 이미지 resize 
from glob import glob # 파일 검색 패턴 사용(문자열 경로, * 사용) 

img_resize = [] 

for file in glob(path + '/data/*.jpg'): # jpg 파일 검색    
    #print(file) # jpg 파일 경로
    img = Image.open(file) # image read 
    
    img_re = img.resize( (150, 120) ) # image resize
        # 배열로 바꿔 리스트에 저장한다. 
    img_resize.append(np.array(img_re)) # numpy array 

print(img_resize)

# list -> array 
img_resize_arr = np.array(img_resize) 

img_resize_arr.shape # (3, 120, 150, 3) : 4d(size, h, w, c)

for i in range(3) :
    plt.imshow(X=img_resize_arr[i])
    plt.show()
