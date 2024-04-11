# -*- coding: utf-8 -*-
"""
Pandas 객체 시각화 : 연속형 변수 시각화  
 - hist, kde, scatter, box 등 

"""

import pandas as pd
import numpy as np # dataset 
import matplotlib.pyplot as plt # chart

# file 경로 
path = r'C:\ITWILL\4_Python_ML\data'

# 1. 산점도 
dataset = pd.read_csv(path + '/dataset.csv')
print(dataset.info())
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 217 entries, 0 to 216
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   resident  217 non-null    int64  
 1   gender    217 non-null    int64  
 2   job       205 non-null    float64
 3   age       217 non-null    int64  
 4   position  208 non-null    float64
 5   price     217 non-null    float64
 6   survey    217 non-null    int64  
dtypes: float64(3), int64(4)
memory usage: 12.0 KB
None
'''
# 연속형 변수 
plt.scatter(dataset['age'], dataset['price'], c=dataset['gender'])
plt.show()


# 2. hist, kde, box
# DataFrame 객체 
df = pd.DataFrame(np.random.randn(100, 4), # [100 rows x 4 columns]
               columns=('one','two','three','fore'))

# 1) 히스토그램
df['one'].plot(kind = 'hist', title = 'histogram')
plt.show()

# 2) 커널밀도추정 
# Kernel Density Estimation, KDE
# 데이터의 분포를 부드럽고 연속적인 곡선으로 표현
# 비모수적 방법 ★★★
df['one'].plot(kind = 'kde', title='kernel density plot')
plt.show()

# 3) 박스플롯
df.plot(kind='box', title='boxplot chart')
plt.show()


# 3. 산점도 행렬 
from pandas.plotting import scatter_matrix

# 3) iris.csv
iris = pd.read_csv(path + '/iris.csv')

cols = list(iris.columns)

x = iris[cols[:4]] 

# 산점도 matrix 
scatter_matrix(x)
plt.show()


