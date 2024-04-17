# -*- coding: utf-8 -*-
"""
step01_datasets_linearRegression.py

sklearn 패키지 
 - python 기계학습 관련 도구 제공 
 

metrics 모듈에서...
 - mean_squared_error(y_true, y_pred)
 - r2_score(y_true, y_pred)

sklearn의 모델 내장메소드로.. 
 - model.score(X=X_train, y=y_train)
 - model.score(X=X_test, y= y_test)
"""

from sklearn.datasets import load_diabetes # dataset 
from sklearn.linear_model import LinearRegression # model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  


# 1. dataset load (당뇨병 데이터셋)
diabetes = load_diabetes() # 객체 반환  
help(load_diabetes()) 
X, y = load_diabetes(return_X_y = True) # X변수, y변수 반환 
dir(diabetes)
'''
['DESCR',
 'data',
 'data_filename',
 'data_module',
 'feature_names',
 'frame',
 'target',
 'target_filename']
'''

# 2. X, y변수 특징 
X.mean() # -1.6638274468590581e-16
X.min() # -0.137767225690012
X.max() # 0.198787989657293
X.shape # (442,10)
X.mean(axis = 0 ).shape # (10,)  # axis = 0 : 행축에 대한 통계량 

# y변수 
y.mean() # 152.13348416289594
y.min() # 25.0
y.max() # 346.0


# 3. train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X, y, 
                                    test_size=0.3, 
                                    random_state=123) 
'''
★★★
X_test 를 통해 y het(예측치): y_pred
y_test 는 정답 : y_true
'''
# test_size : 평가셋 비율 
# random_state : 시드값 지정 


# 4. model 생성 
lr = LinearRegression() # model object
model = lr.fit(X=X_train, y=y_train)  

dir(model)
'''
언더바(_) 가 붙는 애들은 '속성'
나머지는 메소드
coef_   # 기울기(회귀계수)
intecept_
predict # 메소드
score  
'''
model.coef_
'''
각 변수들에 대한 회귀계수 (가중치)
array([  10.45319644, -261.16273528,  538.85049356,  280.72085805,
       -855.24407564,  472.1969838 ,  166.53481397,  309.88981052,
        684.06085168,  102.3789942 ])
'''
model.intercept_ # 152.61082386550538

# 5. model 평가  
# 예측값과 실제 정답 비교를 위해 
y_pred = model.predict(X=X_test)  
y_true = y_test  


# 1) 평균제곱오차(MSE)  

MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE)
# >> MSE = 2926.8196257936324
# scailing을 안했기 때문에 값이 크게 나옴!!


# 2) 결정계수 # 1이 젤 좋음
score = r2_score(y_true, y_pred)
print('r2 score =', score) 
# >> r2 score = 0.5078253552814805

# 3) score() 이용
# ★★★ 예측치 넣지 않아도, 데이터셋과 정답셋(y_train, y_test) 만으로도 산출 가능 
model.score(X=X_train, y=y_train) # 훈련셋 : 0.5174981091172836
model.score(X=X_test, y = y_test) # 평가셋 : 0.5078253552814805
