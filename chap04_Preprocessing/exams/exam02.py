# 문2) mtcars 자료를 이용하여 다음과 같은 단계로 이상치를 처리하시오.

import pandas as pd 
import seaborn as sn # 데이터셋 로드 
pd.set_option('display.max_columns', 50) # 최대 50 칼럼수 지정
import matplotlib.pyplot as plt # boxplot 시각화 


# 데이터셋 로드 
data = sn.load_dataset('mpg')
data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   mpg           398 non-null    float64
 1   cylinders     398 non-null    int64  
 2   displacement  398 non-null    float64
 3   horsepower    392 non-null    float64
 4   weight        398 non-null    int64  
 5   acceleration  398 non-null    float64
 6   model_year    398 non-null    int64  
 7   origin        398 non-null    object 
 8   name          398 non-null    object 
dtypes: float64(4), int64(3), object(2)
memory usage: 28.1+ KB
'''
print(data)


# 단계1. boxplot으로 'acceleration' 칼럼 이상치 탐색 
data['acceleration'].plot(kind='box')

# 단계2. IQR 방식으로 이상치 탐색
des = data.describe()
des 
'''
       acceleration  model_year  
count    398.000000  398.000000  
mean      15.568090   76.010050  
std        2.757689    3.697627  
min        8.000000   70.000000  
25%       13.825000   73.000000  
50%       15.500000   76.000000  
75%       17.175000   79.000000  
max       24.800000   82.000000  
'''

# 1) IQR 수식 작성 
Q3 = des.loc['75%','acceleration']
Q1 = des.loc['25%','acceleration']
IQR = Q3- Q1
outlier = 1.5*IQR
maxval = Q3 + outlier
minval = Q1 - outlier

# 2) 이상치 확인 
new_data = data[(data['acceleration']<= maxval) & (data['acceleration'] >= minval)]
new_data['acceleration'].plot(kind='box')

# 단계3. 이상치 대체 : 정상범주의 하한값과 상한값 대체 
new_data.loc[new_data['acceleration']>maxval, 'acceleration'] = maxval
new_data.loc[new_data['acceleration']<minval, 'acceleration'] = minval

# 단계4. boxplot으로 'acceleration' 칼럼 이상치 처리결과 확인 
new_data['acceleration'].plot(kind='box')

new_data[(new_data['acceleration']>maxval) | (new_data['acceleration']<minval)]
'''
Empty DataFrame
'''