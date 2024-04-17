# -*- coding: utf-8 -*-
"""
step04_scaling_LinearRegression.py

특징변수 데이터변환(features scaling) : 이물질 제거 
 1. 특징변수(x변수) : 값의 크기(scale)에 따라 model 영향을 미치는 경우
      ex) 범죄율(-0.01~0.99), 주택가격(99~999)
   1) 표준화 : X변수를 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일 시킴 
      -> 회귀모델, SVM 계열은 X변수가 정규분포라고 가정하에 학습이 진행되므로 표준화를 적용   
   2) 최소-최대 정규화 : 서로 다른 척도(값의 범위)를 갖는 X변수를 대상으로 최솟값=0, 최댓값=1로 통일 시킴 
      -> 트리모델 계열(회귀모델 계열이 아닌 경우)에서 서로 다른 척도를 갖는 경우 적용 

 2. 타깃변수(y변수) : 로그변환(log1p() 함수 이용 ) 
"""

import pandas as pd # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

from sklearn.preprocessing import scale # 표준화(mu=0, st=1) 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
import numpy as np # 로그변환 + 난수 


###############################
### 특징변수(x변수) 데이터변환 
###############################
# 1. dataset load
path = r'C:\ITWILL\4_Python_ML\data'

# - 1978 보스턴 주택 가격에 미치는 요인을 기록한 데이터 
boston = pd.read_csv(path + '/BostonHousing.csv')
boston.info()
'''
 0   CRIM       506 non-null    float64 : 범죄율
 1   ZN         506 non-null    float64 : 25,000 평방피트를 초과 거주지역 비율
 2   INDUS      506 non-null    float64 : 비소매상업지역 면적 비율
 3   CHAS       506 non-null    int64   : 찰스강의 경계에 위치한 경우는 1, 아니면 0
 4   NOX        506 non-null    float64 : 일산화질소 농도
 5   RM         506 non-null    float64 : 주택당 방 수
 6   AGE        506 non-null    float64 : 1940년 이전에 건축된 주택의 비율
 7   DIS        506 non-null    float64 : 직업센터의 거리
 8   RAD        506 non-null    int64   : 방사형 고속도로까지의 거리 
 9   TAX        506 non-null    int64   : 재산세율
 10  PTRATIO    506 non-null    float64 : 학생/교사 비율
 11  B          506 non-null    float64 : 인구 중 흑인 비율
 12  LSTAT      506 non-null    float64 : 인구 중 하위 계층 비율
 13  MEDV       506 non-null    float64 : y변수 : 506개 타운의 주택가격(단위 1,000 달러)
 14  CAT. MEDV  506 non-null    int64   : 사용 안함(제외)
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)

y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 안됨 
X.mean() # 70.07396704469443
y.mean() # 22.532806324110677

# 2. scaling 함수 
def scaling(X, y, kind='none') : # (X, y, '유형')
    # x변수 스케일링  
    if kind == 'minmax_scale' :  
        X_trans = minmax_scale(X) # 1. 정규화
    elif kind == 'scale' : 
        X_trans = scale(X) # 2. 표준화
    elif kind == 'log' :  
        X_trans = np.log1p(np.abs(X)) # 3. 로그변환
    else :
        X_trans = X # 4. 기본 
    
    # y변수 로그변환 
    if kind != 'none' :
        y = np.log1p(y)   
    
    # train/test split 
    X_train,X_test,y_train,y_test = train_test_split(
        X_trans, y, test_size = 30, random_state=1)   
    
    print(f"scaling 방법 : {kind}, X 평균 = {X_trans.mean()}")
    return X_train,X_test,y_train, y_test

# kind = 를 바꿔가며 학습시켜 평가해보자 ★★★★
X_train,X_test,y_train,y_test = scaling(X, y,'scale') 

# 3. 회귀모델 생성하기
model = LinearRegression().fit(X=X_train, y=y_train) # 지도학습 


# 4. model 평가하기
model_train_score = model.score(X_train, y_train) 
model_test_score = model.score(X_test, y_test)
print('model train score =', model_train_score)
print('model test score =', model_test_score)
'''
(스케일링 none 일때)
model train score = 0.7410721208614652
model test score = 0.717046343087048
(y변수 log변환 했을때)
model train score = 0.7971243680501753
model test score = 0.8217362767578847
(MinMax 스케일링_ 정규화했을 때)
model train score = 0.7910245321591743
model test score = 0.7633961405434471
(표준화했을 때)
model train score = 0.7910245321591743
model test score = 0.763396140543447

'''

y_pred = model.predict(X_test)
y_true = y_test
print('R2 score =',r2_score(y_true, y_pred)) # model test score 동일 
mse = mean_squared_error(y_true, y_pred)
print('MSE =', mse)
'''
(스케일링 none 일때)
R2 score = 0.717046343087048
MSE = 20.20083182974836

(y변수 로그변환했을 때)
R2 score = 0.8217362767578847
MSE = 0.021037701520680095  << 잔차값이 줄어든 것을 볼 수 있다. 

(MinMax 스케일링_ 정규화했을 때)
R2 score = 0.7633961405434471
MSE = 0.027922682660046293

(표준화했을 때)
R2 score = 0.763396140543447
MSE = 0.02792268266004631

'''