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
# ★객체 이름이 칼럼 이름이다. 
# 시리즈 객체 만들고 데이터 프레임에 추가할 수 있음. 
# 방법 1) list 이용 
price = pd.Series([3000,2000,1000,5200]) 
print(price)

fruit = Series([6000, 3500, 1500, 5000], 
                  index=['apple', 'mango','orange', 'kiwi'])

# 방법 2) dict 이용 
fruit = Series({'apple':6000, 'mango':3500,'orange':1500,'kiwi':5000}) 
print(fruit)
fruit.shape # (4,)
fruit['kiwi'] # 5000

# 3) 조건식 
price[price > 3000] # >> 3  5200 

# 2. indexing/slicing 
ser = Series([4, 4.5, 6, 8, 10.5])  
print(ser)

# list와 동일한 방식 
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
# 같은 인덱스끼리 연산한다. 
# 결측치는 연산 대상에서 제외된다. 
s3 = s1 + s2
print(s3)
'''
a    7000.0
b       NaN
c    4500.0
d    3500.0
dtype: float64
'''

# 결측치 처리
'''
fillna(): 결측치 채우기 
결측치 유/무 확인 후 True/False 반환 
    notnull(): 결측치 아닌 것 
    isna() or isnull() : 결측치인 것 
.isnull().sum() : 결측치 갯수 반환
'''
result = s3.fillna(s3.mean())
# 평균값으로 결측치 채우기
print(result)

result2 = s3.fillna(0)
print(result2)


# 결측치 확인
# notnull() 
# 방법 1. T/F 반환
result3 = s3.notnull() # <=> pd.notnull(s3)
print(result3)
'''
a     True
b    False
c     True
d     True
dtype: bool
'''
print(s3[result3]) # True인 데이터 반환 
'''
a    7000.0
c    4500.0
d    3500.0
dtype: float64
'''
# 방법 2. 
r = pd.notnull(s3)
print(r)
result4 = s3[pd.notnull(s3)] # True인 데이터 반환 
print(result4)

'''
a    7000.0
c    4500.0
d    3500.0
dtype: float64
'''
# isnull() 
result5 = s3.isnull()
print(result5)
'''
a    False
b     True
c    False
d    False
dtype: bool
'''

# 4. Series 연산 

# 1) 범위 설정해 값 수정하기 
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

# 유일값 
ser.unique() 
# 출현 빈도수 
ser.value_counts() 
