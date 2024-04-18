# -*- coding: utf-8 -*-
"""
<로짓변환 과정>
# 단계1 : 오즈비(Odds ratio) : 실패(0)에 대한 성공(1) 비율
         (0 ~ Inf)  odds_ratio = p / (1-p) 
# 단계2 : 로짓변환 = log(오즈비)
         (-Inf ~ +Inf)  logit1 = log(odds_ratio) 
# 단계3 : sigmoid 함수 : 로짓값 -> 확률값
         (0 ~ 1)  함수의 x에 logit값이 들어간다

로지스틱회귀모델을 이해를 위한 오즈비(odds_ratio)와 로짓변환 그리고 시그모이드함수    
 - 오즈비(odds_ratio) : 실패에 대한 성공확률 
 - 로짓값(logit value) : 오즈비를 로그변환한 값 = log(오즈비) 
 - 시그모이드(sigmoid) : 로짓값을 0 ~ 1 사이 확률로 변환하는 함수      
"""

import numpy as np


# sigmoid 함수 정의 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# [실습] 오즈비(odds_ratio)와 로짓값 그리고 sigmoid함수 

# 1) 성공확률 50% 미만
p = 0.2   

odds_ratio = p / (1-p) # 오즈비(odds_ratio)
# 오즈비가 1 미만이면 x가 감소방향으로 y에 영향
odds_ratio # 0.25
logit = np.log(odds_ratio) # 로짓값   
logit # -1.3862943611198906
sig = sigmoid(logit) # sigmoid함수
sig # 0.2
y_pred = 1 if sig > 0.5 else 0 
y_pred# 0

# 2) 성공확률 50% 이상
p = 0.6 
  
odds_ratio = p / (1-p) # 오즈비(odds_ratio)
# 오즈비가 1 이상이면 x가 증가방향으로 y에 영향
logit = np.log(odds_ratio) # 로짓값   
sig = sigmoid(logit) # sigmoid함수



###########################################
### 통계적인 방법의 로지스틱회귀모델 
###########################################

import pandas as pd


path = r'C:\ITWILL\4_Python_ML\data'

skin = pd.read_csv(path + '/skin.csv')
skin.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   cust_no      30 non-null     int64  : 제외 
 1   gender       30 non-null     object : x변수
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object : y변수
'''

# 1. X, y변수 인코딩
X = skin.drop(['cust_no','cupon_react'], axis = 1)  

# (명목척도) X변수 인코딩 
X = pd.get_dummies(X, columns=['gender', 'job' , 'marry', 'car'],
                   drop_first=True, dtype='uint8') 

# (명목척도) y변수 인코딩 
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(skin.cupon_react)

new_skin = X.copy() 
new_skin['y'] = y 
new_skin.info()
'''
 0   age          30 non-null     int64
 1   gender_male  30 non-null     uint8
 2   job_YES      30 non-null     uint8
 3   marry_YES    30 non-null     uint8
 4   car_YES      30 non-null     uint8
 5   y            30 non-null     int32
'''


# 2. 상관계수 행렬 : 0.25 미만 변수 제외 
corr = new_skin.corr()
corr
'''
                      age   gender_male  ...   car_YES         y
age          1.000000e+00  8.275114e-18  ...  0.092110  0.276329
gender_male  8.275114e-18  1.000000e+00  ... -0.027462 -0.302079
job_YES      1.842190e-01  2.746175e-02  ... -0.049774  0.221719
marry_YES   -1.936492e-01 -2.886751e-01  ... -0.095130  0.475651
car_YES      9.210952e-02 -2.746175e-02  ...  1.000000  0.185520
y            2.763285e-01 -3.020793e-01  ...  0.185520  1.000000

[6 rows x 6 columns]
'''
corr = corr['y'] # series
lst_corr= []
for i, v in corr.items():
    if v >=0.2 or v <= -0.2 : 
        lst_corr.append(i)
lst_corr
new_skin = new_skin[lst_corr]

# 3. 로지스틱회귀모델 : formula 형식 
from statsmodels.formula.api import logit

formula = logit(formula='y ~ age + gender_male + marry_YES', data = new_skin)
# 종속변수 ~ 독립변수1 + 독립변수2 +...

model = formula.fit()
dir(model)
'''
logit객체 메소드
fittedvalues :적합치
params : 회귀계수
summary() : 분석결과 
'''
y = new_skin.y # 종속변수 
y_fitted = model.fittedvalues # model 적합치(예측치) # 로짓값      

# 로짓값 => 확률(sigmoid func)
y_sig = sigmoid(y_fitted)
y_sig
# 확률 => 0 or 1 로 변환 
y_pred = [1 if y > 0.5 else 0 for y in y_sig]
y_pred # 종속변수 예측값

result = pd.DataFrame({'y': y, 'y_sig':y_sig, 'y_pred':y_pred})
result
'''
    y     y_sig  y_pred
0   0  0.453130       0
1   0  0.310961       0
2   0  0.310961       0
3   0  0.332939       0
4   0  0.792961       1
5   0  0.055545       0
...
25  0  0.453130       0
26  1  0.792961       1
27  0  0.332939       0
28  1  0.875498       1
29  1  0.970153       1
'''