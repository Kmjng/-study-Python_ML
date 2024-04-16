# -*- coding: utf-8 -*-
"""
★★★★ 독립변수들 표준화 스케일링 하기 ★★★★
문4) BostonHousing 데이터셋을 대상으로 다음과 같은 단계별로 전처리를 수행하시오. 
"""

import pandas as pd # csv file load 
from sklearn.preprocessing import StandardScaler # 스케일링 도구
import numpy as np # np.log1p() 함수 

# 단계1. dataset load

### Boston 주택가격 
boston = pd.read_csv(r'C:/ITWILL/4_Python_ML/data/BostonHousing.csv')
boston.info()
'''
RangeIndex: 506 entries, 0 to 505
Data columns (total 15 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   CRIM       506 non-null    float64  -> 1번 x변수  
 1   ZN         506 non-null    float64
 2   INDUS      506 non-null    float64
 3   CHAS       506 non-null    int64  
 4   NOX        506 non-null    float64
 5   RM         506 non-null    float64
 6   AGE        506 non-null    float64
 7   DIS        506 non-null    float64
 8   RAD        506 non-null    int64  
 9   TAX        506 non-null    int64  
 10  PTRATIO    506 non-null    float64
 11  B          506 non-null    float64
 12  LSTAT      506 non-null    float64 -> 13번 x변수 
 13  MEDV       506 non-null    float64 -> y변수
 14  CAT. MEDV  506 non-null    int64   -> 변수 제거   
'''
 
# 단계2 :  'CAT. MEDV' 변수 제거 후 new_df 만들기 
new_df = boston.drop('CAT. MEDV', axis = 1) 
new_df.info()
# 단계3 : new_df에서 1~13칼럼으로 X변수 만들기 
X = new_df.drop('MEDV', axis=1)
'''
방법2. 
x_cols = list(new_df.columns)
X= new_df[X_cols[:13]]  
방법3.
new_df.iloc[:,:13]

'''

# 단계4 : new_df에서 'MEDV' 칼럼으로 y변수 만들기 
y = new_df.MEDV
new_df['y']= y
X
# 단계5 : X변수와 y변수 요약통계량 확인하기   
X.describe().T # 모든 변수에 대해 보기 위해 Transpose 
# 평균(mean) 확인을 하고 scaling 결정
'''
         count        mean         std  ...        50%         75%       max
CRIM     506.0    3.613524    8.601545  ...    0.25651    3.677083   88.9762
ZN       506.0   11.363636   23.322453  ...    0.00000   12.500000  100.0000
INDUS    506.0   11.136779    6.860353  ...    9.69000   18.100000   27.7400
CHAS     506.0    0.069170    0.253994  ...    0.00000    0.000000    1.0000
NOX      506.0    0.554695    0.115878  ...    0.53800    0.624000    0.8710
RM       506.0    6.284634    0.702617  ...    6.20850    6.623500    8.7800
AGE      506.0   68.574901   28.148861  ...   77.50000   94.075000  100.0000
DIS      506.0    3.795043    2.105710  ...    3.20745    5.188425   12.1265
RAD      506.0    9.549407    8.707259  ...    5.00000   24.000000   24.0000
TAX      506.0  408.237154  168.537116  ...  330.00000  666.000000  711.0000
PTRATIO  506.0   18.455534    2.164946  ...   19.05000   20.200000   22.0000
B        506.0  356.674032   91.294864  ...  391.44000  396.225000  396.9000
LSTAT    506.0   12.653063    7.141062  ...   11.36000   16.955000   37.9700

[13 rows x 8 columns]
'''
y.describe()
'''
count    506.000000
mean      22.532806
std        9.197104
min        5.000000
25%       17.025000
50%       21.200000
75%       25.000000
max       50.000000
Name: MEDV, dtype: float64
'''

# 단계6. X변수 표준화 : X변수 표준화 후 칼럼명을 지정하여 new_df2 만들기  
scaler = StandardScaler()  # 1~13칼럼 표준화
X_scaled = scaler.fit_transform(X=X) 
X_scaled.shape # >> (506,13)
X_scaled.describe() # AttributeError

new_df2 = pd.DataFrame(X_scaled) 
new_df2.describe().T # columns이 0,....12임 
new_df2 = pd.DataFrame(X_scaled , columns = X.columns) # 열이름
new_df2.describe().T 
'''
         count          mean      std  ...       50%       75%       max
CRIM     506.0 -1.123388e-16  1.00099  ... -0.390667  0.007397  9.933931
ZN       506.0  7.898820e-17  1.00099  ... -0.487722  0.048772  3.804234
INDUS    506.0  2.106352e-16  1.00099  ... -0.211099  1.015999  2.422565
...
TAX      506.0  0.000000e+00  1.00099  ... -0.464673  1.530926  1.798194
PTRATIO  506.0 -4.212704e-16  1.00099  ...  0.274859  0.806576  1.638828
B        506.0 -7.442444e-16  1.00099  ...  0.381187  0.433651  0.441052
LSTAT    506.0 -3.089316e-16  1.00099  ... -0.181254  0.603019  3.548771

[13 rows x 8 columns]
'''
# 단계7. y변수 로그화  : y변수 로그변환 후 new_df2에 'MEDV'이름으로 칼럼 추가하기   
y= np.log1p(y)
new_df2['MEDV'] = y 

new_df2.describe().T
'''
....
PTRATIO  506.0 -4.212704e-16  1.000990  ...  0.274859  0.806576  1.638828
B        506.0 -7.442444e-16  1.000990  ...  0.381187  0.433651  0.441052
LSTAT    506.0 -3.089316e-16  1.000990  ... -0.181254  0.603019  3.548771
MEDV     506.0  3.085437e+00  0.386966  ...  3.100092  3.258097  3.931826

[14 rows x 8 columns]
'''
# 단계8 : 최종결과 완성된 new_df2을 대상으로 요약통계량 확인하기(출력결과 참고) 
'''
               CRIM            ZN  ...         LSTAT        MEDV
count  5.060000e+02  5.060000e+02  ...  5.060000e+02  506.000000
mean  -8.513173e-17  3.306534e-16  ... -1.595123e-16    3.085437
std    1.000990e+00  1.000990e+00  ...  1.000990e+00    0.386966
min   -4.197819e-01 -4.877224e-01  ... -1.531127e+00    1.791759
25%   -4.109696e-01 -4.877224e-01  ... -7.994200e-01    2.891757
50%   -3.906665e-01 -4.877224e-01  ... -1.812536e-01    3.100092
75%    7.396560e-03  4.877224e-02  ...  6.030188e-01    3.258097
max    9.933931e+00  3.804234e+00  ...  3.548771e+00    3.931826
'''

