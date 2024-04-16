# -*- coding: utf-8 -*-
"""
reshape : 모양 변경 (크기 변경 불가★)
 - 1차원 -> 2차원 
 - 2차원 -> 다른 형태의 2차원  
T : 전치행렬 
swapaxis : 축 변경 
transpose : 축 번호 순서로 구조 변경 
"""

import numpy as np

# 1. 모양변경(reshape)
lst = list(range(1, 13)) # 1차원 배열
 
arr2d = np.array(lst).reshape(3, 4) # 모양변경
print(arr2d)
# 주의 : 길이(size) 변경 불가 
'''
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
'''

# 2. 전치행렬
# 2차원 : 전치행렬
print(arr2d.T)
'''
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
'''
print(arr2d.T.shape) # (4, 3)

# 3. swapaxes : 축 변경 
print('swapaxes')
print(arr2d.swapaxes(0, 1)) 
'''
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
'''

# 4. transpose
# 3차원 : 축 순서(0,1,2)를 이용하여 자료 구조 변경 
arr3d = np.arange(1, 25).reshape(4, 2, 3)#(4면2행3열)
print(arr3d)
'''
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]

 [[13 14 15]
  [16 17 18]]

 [[19 20 21]
  [22 23 24]]]
'''
print(arr3d.shape) # (4, 2, 3)

# default : (면,행,열) -> (열,행,면) 거꾸로
arr3d_def = arr3d.transpose() # 0:면,1:행,2:열
print(arr3d_def.shape) # (3, 2, 4)
print(arr3d_def)
'''
[[[ 1  7 13 19]
  [ 4 10 16 22]]

 [[ 2  8 14 20]
  [ 5 11 17 23]]

 [[ 3  9 15 21]
  [ 6 12 18 24]]]
'''

# user : (면,행,열) -> (열,면,행)

arr3d_user = arr3d.transpose(2,0,1) # 0:면,1:행,2:열
arr3d_user.shape # (3, 4, 2)
print(arr3d_user)
'''
[[[ 1  4]
  [ 7 10]
  [13 16]
  [19 22]]

 [[ 2  5]
  [ 8 11]
  [14 17]
  [20 23]]

 [[ 3  6]
  [ 9 12]
  [15 18]
  [21 24]]]
'''