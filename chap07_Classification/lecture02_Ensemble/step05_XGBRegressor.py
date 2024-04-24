# -*- coding: utf-8 -*-
"""
 - XGBoost 회귀트리 예
"""

from xgboost import XGBRegressor # 회귀트리 모델 
from xgboost import plot_importance # 중요변수 시각화 

import pandas as pd # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# 스케일링 도구 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
import numpy as np # 로그변환 + 난수 



### 1. dataset load & preprocessing 

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
 14  CAT. MEDV  506 non-null    int64   : 제외  
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)
X.mean(axis =0) # 스케일링 할지안할지 
# 최대평균이 408, 최소평균이 0.0 이어서 스케일링이 필요함 
# minmax 스케일링(정규화) 


y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 
X_scaled = pd.DataFrame(minmax_scale(X), columns=X.columns) # 정규화
y = np.log1p(y) # 로그변환 
# 십진수 단위로 구성된 y변수를 로그변환 해준다. ★★★


###  2. train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.3)


### 3. model 생성 : 회귀트리에서의 활성함수 SE 
model = XGBRegressor(objective='reg:squarederror').fit(X=X_train, y=y_train) # objective : 활성함수
print(model)


### 4. 중요변수 확인 
fscore = model.get_booster().get_fscore()
print(fscore)
'''
{'CRIM': 556.0, 'ZN': 33.0, 'INDUS': 70.0, 'CHAS': 14.0, 
 'NOX': 152.0, 'RM': 342.0, 'AGE': 290.0, 'DIS': 241.0, 
 'RAD': 32.0, 'TAX': 69.0, 'PTRATIO': 74.0, 'B': 226.0, 
 'LSTAT': 278.0}
'''
# 중요변수 시각화 : x변수명 없음 
plot_importance(model, max_num_features=13) # 13개까지 나타냄 


### 5. model 평가 
y_pred = model.predict(X = X_val)  # 예측치 

mse = mean_squared_error(y_val, y_pred)
print('MSE =',mse)  # MSE = 0.02119909296604185

score = r2_score(y_val, y_pred)
print('r2 score =', score) # r2 score = 0.8590175164169235


################################
### 탐색적 자료분석 (원본데이터를 갖고 탐색할 것★★★)
import seaborn as sn 
boston.CRIM #연속형 변수 # X
boston.MEDV #연속형 변수 # y

# scatterplot 
sn.scatterplot(data = boston, x= boston.CRIM, y= boston.MEDV)
sn.scatterplot(data = boston, x= boston.RM, y= boston.MEDV)

# 데이터 구간화 ★★★
# 비율척도 -> 명목척도(구간화)
# 방법 (1) 
boston.RM.min() # 3.561
boston.RM.max() # 8.78
'''
3 : 4미만 
4 : 4~5
5 : 5~6
6 : 6~7
7 : 7이상
'''

rm = boston.RM
rm_new = [] 

for r in rm :
    if r < 4 :
        rm_new.append(3)
    elif r > 4 and r < 5 :
        rm_new.append(4)
    elif r > 5 and r < 6 :
        rm_new.append(5)        
    elif r > 6 and r < 7 :
        rm_new.append(6)
    else :
       rm_new.append(7) 
       
rm[:5] # 비율척도       
rm_new[:5] # 구간화       

# 칼럼 추가 
boston['rm_new'] = rm_new

sn.barplot(data = boston, x=boston.rm_new, y = boston.MEDV)
# [해설] 대체적으로 방의 개수가 많을 수록 주택 가격이 높아진다.



###########################################
### 6. model save & Testing 
# 학습된 모델 객체 자체를 피클타입으로 저장한다. 
import pickle # binary file 

# model file save 
pickle.dump(model, open('xgb_model.pkl', mode='wb'))

# model file load 
load_model = pickle.load(open('xgb_model.pkl', mode='rb'))


# final model Test 
idx = np.random.choice(a=len(X_scaled), size=200, replace=False) # test set 만들기 

X_test, y_test = X_scaled.iloc[idx], y[idx]

y_pred = load_model.predict(X = X_test) # new test set (여기서는 이미 학습된거 씀)

score = r2_score(y_test, y_pred)
print('r2 score =', score) 
# r2 score = 0.9720853747378826 
