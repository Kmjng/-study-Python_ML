# -*- coding: utf-8 -*-
"""
문2) 이항검정 : 95% 신뢰수준에서 토요일(Sat)에 오는 여자 손님 중 
비흡연자가 흡연자 보다 많다고 할 수 있는가?

 귀무가설(H0) : 비흡연자와 흡연자의 비율은 차이가 없다.(P=0.5)
"""

from scipy import stats # 이항검정 
import pandas as pd # csv file read

path = r'C:\ITWILL\4_Python_ML\data'
tips = pd.read_csv(path + "/tips.csv")
print(tips.info())
'''
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    object : 성별 
 3   smoker      244 non-null    object : 흡연유무 
 4   day         244 non-null    object : 행사요일 
 5   time        244 non-null    object 
 6   size        244 non-null    int64 
'''
 

# 1. 성별이 여성(Female) 이면서 행사 요일이 토요일(Sat)인 경우만 서브셋 만들기  
subset = tips[(tips['day']=='Sat') & (tips['sex']=='Female')]
subset

# 2. 흡연유무(smoker) 대상으로 흡연자와 비흡연자별 빈도수 : value_counts() 이용 
subset.smoker.value_counts()
'''
smoker
Yes    15
No     13
Name: count, dtype: int64

차이 있냐없냐 => 양측검정 
alpha = 0.05
'''
k = subset.smoker.value_counts()[0]


# 3. 이항검정(binom test)
k = subset.smoker.value_counts()[0] # 성공횟수 15
n = subset.smoker.value_counts().sum() # 시행횟수  28

result = stats.binomtest(k=k, n=n, p=0.5, alternative='two-sided')
# 둘중 하나니까 p =0.5


pvalue = result.pvalue
pvalue 
# 0.8505540192127228

alpha = 0.05
if pvalue > alpha : # 유의확률 > 유의수준 
    print(f"p-value({pvalue}) >= 0.05 : 비흡연자와 흡연자의 비율은 차이가 없다.")
else:
    print(f"p-value({pvalue}) < 0.05 : 비흡연자와 흡연자의 비율은 차이가 있다.")
    
'''
p-value(0.8505540192127228) >= 0.05 : 
    비흡연자와 흡연자의 비율은 차이가 없다.
'''




