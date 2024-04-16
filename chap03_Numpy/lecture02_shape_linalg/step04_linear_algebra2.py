# -*- coding: utf-8 -*-
"""
1. 행렬곱
2. 연립방식의 해 
"""

import numpy as np 

# 1. 행렬곱 : 행렬 vs 행렬 곱셈 연산
'''
  - 행렬의 선형변환(선형결합)  : 차원축소 
  - 회귀방정식 : X변수와 기울기 곱셈 
'''
  

A = np.array([1, 2, 3]).reshape(3,1) # 행렬A
B = np.array([2, 3, 4]).reshape(3,1) # 행렬B

A.shape # (3, 1)
B.shape # (3, 1)


# 1) 행렬내적 (스칼라 반환)
dot1 = A.T.dot(B) 


# 2) 행렬외적 (행렬 반환)
dot2 = A.dot(B.T) 


# 2. 연립방정식 해(Solve a linear matrix equation): np.linalg.solve(a, b)
'''
연립방정식 : 2개 이상의 방정식을 묶어놓은 것
다음과 같은 연립방정식의 해(x, y) 구하기 
3*x + 2*y = 53
-4*x + 3*y = -35
'''

a = np.array([[3, 2], [-4, 3]])
b = np.array([53, -35])

x, y = np.linalg.solve(a, b)

