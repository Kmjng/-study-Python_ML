# -*- coding: utf-8 -*-
"""
step02_DataFrame.py

DataFrame 자료구조 특징 
 - 2차원 행렬구조(DB의 Table 구조와 동일함)
 - 칼럼 단위 상이한 자료형 
"""

import pandas as pd # 별칭 
from pandas import DataFrame 


# 1. DataFrame 객체 생성 

# 1) list와 dict 이용 
names = ['hong', 'lee', 'kim', 'park']
ages = [35, 45, 55, 25]
pays = [250, 350, 450, 250]


# key -> 칼럼명, value -> 칼럼값 
frame = pd.DataFrame({'name':names, 'age': ages, 'pay': pays})

# 객체 정보 
frame.info()


# 2) numpy 객체 이용
import numpy as np

data = np.arange(12).reshape(3, 4) # 1d -> 2d
print(data) 

# numpy -> pandas
frame2 = DataFrame(data, columns=['a','b','c','d'])
frame2


# 2. DF 칼럼 참조 
path = r'C:/ITWILL/4_Python_ML/data' # 경로 지정
emp = pd.read_csv(path + "/emp.csv", encoding='utf-8')
print(emp.info())
print(emp)


# 1) 단일 칼럼을 활용
no = emp.No # 방법1
print(no)
name = emp['Name'] # 방법2
print(name)

# 2) 복수 칼럼  
df = emp[['No','Pay']]


# 3. DataFrame 행열 참조 

# 1) loc 속성 : 명칭 기반 
emp.loc[0, :] # 1행 전체 
emp.loc[0] # 열 생략 가능 
emp.loc[0:2] # 1~3행 전체 

# 2) iloc 속성 : 숫자 위치 기반 
emp.loc[0] # 1행 전체 
emp.iloc[0:2] # 1~2행 전체 
emp.iloc[:,1:] # 2번째 칼럼 이후 연속 칼럼 선택



# 4. subset 만들기 

# 1) 특정 칼럼 선택
subset1 =  emp[['Name', 'Pay']]
print(subset1)

# 2) 특정 행 제외 
subset2 = emp.drop(1) # 2행 제외  
subset2_2 = emp.drop([1,3]) # 2행,4행 제외  
print(subset2)
print(subset2_2)


# 3) 조건식으로 행 선택   
subset3 = emp[emp.Pay >= 350] # 급여 350 이하 제외 
print(subset3)


# 판다스의 논리연산자 이용 : &(and), |(or), ~(not) ★★★
# (기본파이썬에서는 비트연산자에 해당함)
emp[(emp.Pay >= 300) & (emp.Pay <= 400)] # 올바른 표현   
emp[(emp.Pay >= 300) and (emp.Pay <= 400)] # ValueError
# 우선순위: 비교연산자 > 논리연산자

# 4) 칼럼값 이용  
iris = pd.read_csv(path + '/iris.csv')
iris.info()
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
'''
iris
iris.Species.value_counts()
'''Species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64'''

print(iris.Species.unique()) # >> ['setosa' 'versicolor' 'virginica']

# ★★★ Species 칼럼에서 setosa, virginica 값을 갖는 데이터 ★★★
# DF명.column명.isin(['값1','값2'])
subset4 = iris[iris.Species.isin(['setosa', 'virginica'])]
print(subset4)

# 5) columns 이용 : 칼럼이 많은 경우 칼럼명 이용  
iris = pd.read_csv(path + 'iris.csv')

names = list(iris.columns) # 전체 칼럼명 list 반환 
names # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

# 독립변수 선택 
iris_x = iris[names[:4]] #iris[열 리스트]

# 리스트 칼럼 하나 제거하기 #remove()메소드
names.remove('Sepal.Width') # 바로 반영됨
