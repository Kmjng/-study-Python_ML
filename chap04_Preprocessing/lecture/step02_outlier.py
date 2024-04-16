
######################################
### 2. 이상치 처리 
######################################
"""
 이상치(outlier) 처리 : 정상범주에서 벗어난 값(극단적으로 크거나 작은 값) 처리  
  - IQR(Inter Quentile Range) 방식으로 탐색과 처리   
"""

import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\insurance.csv")
data.info()
'''
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
'''
 
# 1. 범주형 이상치 탐색  
data.sex.unique() # ['female', 'male']
data.smoker.unique() # ['yes', 'no']
data.region.unique() # ['southwest', 'southeast', 'northwest', 'northeast']


# 2. 숫자형 변수 이상치 탐색  
des = data.describe()  # ★★★
print(des)
'''
               age          bmi     children       charges
count  1338.000000  1338.000000  1338.000000   1338.000000
mean     39.730194    30.524488     1.094918  13270.422265
std      20.224425     6.759717     1.205493  12110.011237
min      18.000000   -37.620000     0.000000   1121.873900
25%      27.000000    26.220000     0.000000   4740.287150
50%      39.000000    30.332500     1.000000   9382.033000
75%      51.000000    34.656250     2.000000  16639.912515
max     552.000000    53.130000     5.000000  63770.428010

요약 통계량을 통해 
age는 최댓값에 이상치, 
bmi는 최솟값에 이상치, 
charge는 알수없음 (IQR 적용해볼 수 있다. 또는 boxplot)
'''


# 3. boxplot 이상치 탐색 
import matplotlib.pyplot as plt

plt.boxplot(data['age']) # age 이상치  
plt.show()

plt.boxplot(data['bmi']) # bmi 이상치 
plt.show()

plt.boxplot(data['charges']) # charges 이상치 
plt.show()

# 4. 이상치 처리 

# 1) bmi 이상치 제거 
df = data.copy() # 복제

df = df[df['bmi'] > 0]


# 2) age 이상치 대체   
df = data.copy() # 복제

df[df['age'] > 100] # 100세 이상   

# 100세 이상 -> 100세 대체 
df.loc[df.age > 100, 'age'] = 100 # 현재 객체 반영 

# 3) charges 이상치 대체 



# 5. IQR방식 이상치 발견 및 처리 

# 1) IQR방식으로 이상치 발견   
'''
 IQR = Q3 - Q1 : 제3사분위수 - 제1사분위수
 outlier_step = 1.5 * IQR
 정상범위 : Q1 - outlier_step ~ Q3 + outlier_step
'''  

Q3 = des.loc['75%', 'age'] 
Q1 = des.loc['25%', 'age'] 
IQR = Q3 - Q1

outlier_step = 1.5 * IQR # 36.0

minval = Q1 - outlier_step
maxval = Q3 + outlier_step
print(f'minval : {minval}, maxval : {maxval}') 
'''minval : -9.0, maxval : 87.0'''

# 2) 이상치 제거  
df = data.copy() # 복제 

df = df[(df['age'] >= minval) & (df['age'] <= maxval)]

# 나이 시각화 
df['age'].plot(kind='box')

# 이상치 제거 (방법: 3*표준편차) # avg - std*3
'''
하한값 = 평균 - 3*표준편차 
상한값 = 평균 + 3*표준편차

minval과 maxval 설정하고, 범위에 해당하는 관측치만 포함시킨다~
''' 
