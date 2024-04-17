# -*- coding: utf-8 -*-
"""
문2) california 주택가격을 대상으로 다음과 같은 단계별로 선형회귀분석을 수행하시오.
"""

# california 주택가격 데이터셋 
'''
캘리포니아 주택 가격 데이터(회귀 분석용 예제 데이터)

•타겟 변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) 
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''

from sklearn.datasets import fetch_california_housing # dataset load
import pandas as pd # DataFrame 생성 
from sklearn.linear_model import LinearRegression  # model
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import mean_squared_error, r2_score # model 평가 
import matplotlib.pyplot as plt 

# 캘리포니아 주택 가격 dataset load 
california = fetch_california_housing()

X = california.data # X변수 
X.shape # (20640, 8)

# 특징변수(8개)와 타켓변수(MEDV)를 이용하여 DataFrame 생성하기  
cal_df = pd.DataFrame(X, columns=california.feature_names)

cal_df["MEDV"] = california.target # y변수 추가 
print(cal_df.info()) 


# 단계1 : 특징변수(8개)와 타켓변수(MEDV) 선택   
# 이거 굳이 해야하나?? 안함.. 
X = None # 특징변수(8개) 선택 
y = None # 타켓변수(MEDV) 선택 


# 단계2 : 75%(train) vs 25%(val) 비율 데이터셋 split 
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, 
                                                    test_size=0.3, random_state =123)

# 단계3 : 회귀모델 생성
cal_model = LinearRegression().fit(X_train, y_train)

y_pred = cal_model.predict(X_test) # X_test를 통해 예측
y_true = y_test


# 단계4 : 모델 검정(evaluation)  : 과적합(overfitting) 확인  
# 결정계수 확인한다. 
cal_model.score(X=X_train, y=y_train) # 0.604714940991568
cal_model.score(X=X_test, y=y_test)   # 0.6093458386889428

# 단계5 : 모델 평가(test) : 평가방법 : MSE, r2_score - 50% 샘플링 자료 이용
MSE = mean_squared_error(y_true, y_pred)
MSE # 0.5165766683279188
r2_score = r2_score(y_true, y_pred)
r2_score # 0.6093458386889428


# 단계6 : 예측치 100개 vs 정답 100개 비교 시각화 

