# -*- coding: utf-8 -*-
"""
step03_Descriptive.py

1. DataFrame의 요약통계량 
2. 상관계수
"""

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

product = pd.read_csv(path + '/product.csv')


# DataFrame 정보 보기 
product.info()


# 앞부분/뒷부분 관측치 5개 보기 
product.head()
product.tail()

# 1. DataFrame의 요약통계량 
summ = product.describe()
print(summ)

# 행/열 통계
product.shape # (264, 3)
product.sum(axis = 0) # 행축  
product.sum(axis = 1) # 열축 

# 산포도 : 분산, 표준편차 
product.var() # axis = 0
product.std() # axis = 0

# 빈도수 : 집단변수 
product['a'].value_counts()


# 유일값 
product['b'].unique()

# 2. 상관관계 
cor = product.corr()
print(cor) # 상관계수 행렬 
'''
          a         b         c
a  1.000000  0.499209  0.467145
b  0.499209  1.000000  0.766853
c  0.467145  0.766853  1.000000
'''
# 공분산 
cov = product.cov() 
print(cov)
'''          a         b         c
a  0.941569  0.416422  0.375663
b  0.416422  0.739011  0.546333
c  0.375663  0.546333  0.686816
'''