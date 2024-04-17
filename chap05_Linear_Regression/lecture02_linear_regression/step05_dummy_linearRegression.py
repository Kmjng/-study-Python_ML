# -*- coding: utf-8 -*-
"""
step05_dummy_linearRegression.py

 가변수(dummy) 변환 : 명목형(범주형) 변수를 X변수 사용
"""

import pandas as pd # csv file, 가변수 
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 


# 1. csv file load 
path = r'C:\ITWILL\4_Python_ML\data'
insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()



# 2. 불필요한 칼럼 제거 : region
new_df = insurance.drop(['region'], axis= 1)
new_df.info()
'''
 0   age      1338 non-null   int64  
 1   sex      1338 non-null   object  
 2   bmi      1338 non-null   float64
 3   children  1338 non-null   int64
 4   smoker   1338 non-null   object  
 5   charges  1338 non-null   float64 -> y변수 
'''
new_df.shape


# 3. X, y변수 선택 
X = new_df.drop('charges', axis= 1)
X.shape #  (1338, 4)

y = new_df['charges']


# 4. 명목형(범주형) 변수 -> 가변수(dummy) 변환 : k-1개 
X.info()
X_dummy = pd.get_dummies(X, columns=['sex', 'smoker'],
               drop_first=True, dtype='uint8')

X_dummy.info()


# 5. 이상치 확인  
X_dummy.describe().T
'''
             count       mean        std  ...      50%       75%     max
age         1338.0  39.730194  20.224425  ...  39.0000  51.00000  552.00
bmi         1338.0  30.524488   6.759717  ...  30.3325  34.65625   53.13
children    1338.0   1.094918   1.205493  ...   1.0000   2.00000    5.00
sex_male    1338.0   0.505232   0.500160  ...   1.0000   1.00000    1.00
smoker_yes  1338.0   0.204783   0.403694  ...   0.0000   0.00000    1.00

[5 rows x 8 columns]
'''

# 6. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_dummy, y, test_size=0.3, random_state=123)


# 7. model 생성 & 평가 
model = LinearRegression().fit(X=X_train, y=y_train)

model.score(X=X_train, y=y_train) # 0.6837522970202745
model.score(X=X_test, y=y_test)   # 0.7236336310934985
'''
 train 데이터에 이상치가 있을 경우 test 데이터보다 결정계수가 낮을 수 있다. 
 그래서? 이상치 전처리를 해줘야 한다 ★★★
'''

# 이상치 처리하기 
X_new = X_dummy[(X_dummy.age > 0) & (X_dummy.age <= 100)] #age 이상치 제거(제외)
X_new = X_new[X_new.bmi > 0]  # bmi 이상치 제거 
X_new.shape # >> (1332,5)
# ★★★★ 주의 ★★★★ 
# 이상치에 해당하는 y 관측치도 삭제해줘야한다.

y = y[X_new.index]  

y.shape # >> (1338,) => (1332,)로 변경됨
y[10:20]
'''
10     2721.32080
11    27808.72510
13    11090.71780
14    39611.75770
15     1837.23700  << 16 이 제거된 것을 확인할 수 있음
17     2395.17155
18    10602.38500
19    36837.46700  ...
'''

# 삭제된 index를 확인해볼 수도 있다. 
X_dummy[~((X_dummy.age > 0) & (X_dummy.age <= 100))].index # ~ 틸다 기호
X_dummy[X_dummy.bmi< 0].index 
# >> [12, 114, 180] , [16, 48, 82]


X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.3, random_state=123)