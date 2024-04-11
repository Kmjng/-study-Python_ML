# -*- coding: utf-8 -*-
"""
Pandas 객체 시각화 : 이산형 변수 시각화 

ex) object.plot(kind='유형',속성)
    object : Series, DataFrame 
    kind : bar, barh, scatter, hist 
    속성 : 제목, 축 이름 등 
 
"""

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

# 1. 기본 차트 시각화 

# 1) Series 객체 시각화 
ser = pd.Series(np.random.randn(10),
          index = np.arange(0, 100, 10))

ser.plot() # 선 그래프 
plt.show()

# 2) DataFrame 객체 시각화
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['one','two','three','fore'])

# 기본 차트 : 선 그래프 
df.plot()  
plt.show()

# 막대차트 
df.plot(kind = 'bar', title='bar chart')
plt.show()


# 2. dataset 이용 
path = r'C:/ITWILL/4_Python_ML/data'

tips = pd.read_csv(path + '/tips.csv')
tips.info()
tips
'''     total_bill   tip     sex smoker   day    time  size
0         16.99  1.01  Female     No   Sun  Dinner     2
1         10.34  1.66    Male     No   Sun  Dinner     3
2         21.01  3.50    Male     No   Sun  Dinner     3
...
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2

[244 rows x 7 columns]
'''
# 행사 요일별 : 파이 차트 
cnt = tips['day'].value_counts()
cnt.plot(kind = 'pie')
plt.show()


# 요일(day) vs 규모(size) : 교차분할표 
# 카이제곱검정 도구 
# 빈도수 
table = pd.crosstab(index=tips['day'], columns=tips['size'])

table 
'''
size  1   2   3   4  5  6
day                      
Fri   1  16   1   1  0  0
Sat   2  53  18  13  1  0
Sun   0  39  15  18  3  1
Thur  1  48   4   5  1  3
'''
type(table) # >>pandas.core.frame.DataFrame


# size : 2~5 칼럼으로 subset 
new_table = table.iloc[:,2:5]
new_table.plot(kind='barh', stacked = True)
plt.title('day vs size')
