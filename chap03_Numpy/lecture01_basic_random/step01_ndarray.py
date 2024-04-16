# -*- coding: utf-8 -*-
"""
Numpy 패키지 
  - 수치 과학용 데이터 처리 목적으로 사용 
  - 선형대수(벡터, 행렬) 연산 관련 함수 제공 
  - N차원 배열, 선형대수 연산, 고속 연산  
  - 수학/통계 함수 제공 
  - indexing/slicing   
  - broadcast 연산
"""

import numpy as np # 별칭 


# 1. list 배열 vs numpy 다차원 배열 

# 1) list 배열
lst = [1, 2, 3, 3.5] # 정수와 실수 자료형
print(lst) 
print(lst * 3 ) # 3번 반복
sum(lst) # 외부 함수 

# 2) numpy 다차원 배열 
arr = np.array(lst) # array([list])  
print(arr) 
print(arr * 0.5) # broadcast 연산 
arr.sum() # 자체 제공 


# 2. array() : 다차원 배열 생성 

lst1 = [3, 5.2, 4, 7]
print(lst1) # 단일 리스트 배열 

arr1d = np.array(lst1) # array(단일list)
print(arr1d.shape) # 자료구조 확인

print('평균 =', arr1d.mean()) 
print('분산=', arr1d.var())
print('표준편차=', arr1d.std()) 



# 3. broadcast 연산 
# - 작은 차원이 큰 차원으로 늘어난 후 연산 

# scalar(0) vs vector(1)
print(0.5 * arr1d)


'''
broadcast 연산 예 
모집단 분산 = sum((x-mu)**2) / n
표본 분산 = sum((x-x_bar)**2) / n-1
'''
x = arr1d # 객체 복제 
mu = x.mean() # 평균  
var = sum((x - mu)**2) / len(x)



# 4. zeros or ones 
zerr = np.zeros( (3, 10) ) # 3행5열 
print(zerr) 

onearr = np.ones( (3, 10) ) # 3행5열 
print(onearr)


# 5. arange 
arr = np.arange(-1.2, 5.5) # float 사용 가능, 배열 객체  
print(arr) # [-1.2 -0.2  0.8  1.8  2.8  3.8  4.8]

# ex) x의 수열에 대한 2차 방정식 
x = np.arange(-1.0, 2, 0.1)
y = x**2 + 2*x + 3 # f(x) 함수 
print(y)

