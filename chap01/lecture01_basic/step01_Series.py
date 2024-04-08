# -*- coding: utf-8 -*-
"""
step01_Series.py

Series 객체 특징 
 - pandas 1차원(vector) 자료구조 
 - DataFrame의 칼럼 구성요소 
 - 수학/통계 관련 함수 제공 
 - indexing/slicing 기능 
"""

import pandas as pd  
from pandas import Series 


# 1. Series 객체 생성 

# 1) list 이용 
price = pd.Series([3000,2000,1000,5200]) 
print(price)


# 2) dict 이용 
person = Series({'name':'홍길동', 'age':35, 'addr':'서울시'}) 
print(person)


# 2. indexing/slicing 
ser = Series([4, 4.5, 6, 8, 10.5])  
print(ser)

ser[:] # 전체 원소 
ser[0] # 1번 원소 
ser[:3] # start~2번 원소 
ser[3:] # 3~end 원소 


# 3. Series 결합과 NA 처리 
s1 = pd.Series([3000, None, 2500, 2000],
               index = ['a', 'b', 'c', 'd'])

s2 = pd.Series([4000, 2000, 3000, 1500],
               index = ['a', 'c', 'b', 'd'])


# Series 결합(사칙연산)
s3 = s1 + s2
print(s3)


# 결측치 처리
result = s3.fillna(s3.mean())
print(result)

result2 = s3.fillna(0)
print(result2)


# 결측치 제거 
result3 = s3[pd.notnull(s3)]
print(result3)


# 4. Series 연산 

# 1) 범위 수정 
print(ser)
ser[1:4] = 5.0


# 2) broadcast 연산 
print(ser * 0.5) 

# 3) 수학/통계 함수 
ser.mean() # 평균
ser.sum() # 합계
ser.var() #  분산
ser.std() # 표준편차
ser.max() # 최댓값
ser.min() # 최솟값
