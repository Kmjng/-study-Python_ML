'''   
lecture02 > step01 관련문제

문6) iris.csv 파일을 읽어와서 다음과 같이 처리하시오.
   <단계1> 1~4 칼럼 대상 vector 생성(col1, col2, col3, col4)    
   <단계2> 1,4 칼럼 대상 합계, 평균, 표준편차 구하기 
   <단계3> 1,2 칼럼과 3,4 칼럼을 대상으로 각 df1, df2 데이터프레임 생성
   <단계4> df1과 df2 칼럼 단위 결합 iris_df 데이터프레임 생성      
'''

import pandas as pd

path = r'C:\ITWILL\4_Python_ML\data'

iris = pd.read_csv(path + '/iris.csv')
print(iris.info())
'''
<class 'pandas.core.frame.DataFrame'>
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
None
'''

# <단계1> 1~4 칼럼 대상 vector 생성(col1, col2, col3, col4)    
col1 = iris['Sepal.Length']
col2 = iris['Sepal.Width']
col3 = iris['Petal.Length']
col4 = iris['Petal.Width']

# <단계2> 1,4 칼럼 대상 합계, 평균, 표준편차 구하기
df_S = pd.DataFrame({'statistics of Sepal.Length':[col1.sum(axis=0),col1.mean(axis=0), col1.std(axis=0)]
                }, index=['sum','mean','std']) 
df_S
'''      statistics of Sepal.Length
sum                   876.500000
mean                    5.843333
std                     0.828066
'''
df_P = pd.DataFrame({'statistics of Petal.Length':[col4.sum(axis=0),col4.mean(axis=0), col4.std(axis=0)]
                }, index=['sum','mean','std']) 
df_P
'''      statistics of Petal.Length
sum                   179.900000
mean                    1.199333
std                     0.762238
'''
# <단계3> 1,2 칼럼과 3,4 칼럼을 대상으로 각 df1, df2 데이터프레임 생성
df1 = pd.DataFrame({'Sepal.Length':col1, 'Sepal.Width':col2})
df2 = pd.DataFrame({'Petal.Length':col1, 'Petal.Width':col2})

# <단계4> df1과 df2 칼럼 단위 결합 iris_df 데이터프레임 생성
iris_df = pd.concat(objs= [df1, df2], axis = 1)
iris_df
'''     Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
0             5.1          3.5           5.1          3.5
1             4.9          3.0           4.9          3.0
2             4.7          3.2           4.7          3.2
3             4.6          3.1           4.6          3.1
4             5.0          3.6           5.0          3.6
..            ...          ...           ...          ...
145           6.7          3.0           6.7          3.0
146           6.3          2.5           6.3          2.5
147           6.5          3.0           6.5          3.0
148           6.2          3.4           6.2          3.4
149           5.9          3.0           5.9          3.0

[150 rows x 4 columns]
'''
