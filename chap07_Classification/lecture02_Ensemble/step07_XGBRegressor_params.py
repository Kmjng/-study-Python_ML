# -*- coding: utf-8 -*-
"""
1. XGBoost Hyper Parameter
2. 학습조기종료(early_stopping)
3. Best Parameter
"""
from xgboost import XGBRegressor # model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd # dataset
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
 14  CAT. MEDV  506 non-null    int64   : 제외  
'''

X = boston.iloc[:, :13] # 독립변수 
X.shape # (506, 13)

y = boston.MEDV # 종속변수 
y.shape #(506,)

# x,y변수 스케일링 안됨 
X.mean() # 70.07396704469443
y.mean() # 22.532806324110677


# 스케일링 
X_scaled = pd.DataFrame(minmax_scale(X), columns=X.columns) # 정규화
y = np.log1p(y) # 로그변환    


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=123)


#################################
## XGBoost Hyper Parameter
#################################
model = XGBRegressor(colsample_bytree=1,
                    learning_rate=0.3, 
                    max_depth=6, 
                    min_child_weight=1,
                    n_estimators=400) # objective='reg:squarederror'
'''
colsample_bytree=1 : 트리를 생성할때 훈련셋에서 feature 샘플링 비율(보통 :0.6~0.9)
learning_rate=0.1 : 학습율(보통 : 0.01~0.2)
max_depth=3 : tree 깊이, 과적합 영향
min_child_weight=1 : 최소한의 자식 노드 가중치 합(자식 노드 분할 결정), 과적합 영향
# - 트리 분할 단계에서 min_child_weight 보다 더 적은 노드가 생성되면 트리 분할 멈춤
n_estimators=100 : tree model 수 
objective='reg:squarederror'
'''

#################################
## 학습조기종료(early_stopping)
#################################
evals = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric='rmse',  # 평가 방법 
                early_stopping_rounds=100, eval_set=evals, verbose=True)
'''
early_stopping_rounds=100 : 조기종료 파라미터
 - 100개 tree model 학습과정에서 성능평가 지수가 향상되지 않으면 조기종료
'''

# 4. model 평가 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('mse =', mse) # mse = 0.02653424888554659

score = r2_score(y_test, y_pred)
print('score=', score)  # score= 0.8132951769205492


#################################
## Best Parameter
#################################
from sklearn.model_selection import GridSearchCV

# 기본 모델 객체 생성 : default parameter 
model = XGBRegressor(n_estimators=100,                  
                    objective='reg:squarederror')

params = {'max_depth':[3, 5, 7], 'min_child_weight':[1, 3],
          'n_estimators':[100, 150, 200], 
          'colsample_bytree':[0.5, 0.7], 'learning_rate':[0.01, 0.5, 0.1]}


# GridSearch model  
grid_model = GridSearchCV(estimator=model, param_grid=params, cv=5)

# GridSearch model 학습 : 훈련셋  
grid_model.fit(X=X_train, y= y_train)

print('best score =', grid_model.best_score_) 
print('best parameter =', grid_model.best_params_)

'''
best score = 0.8728546505525833
best parameter = 
{'colsample_bytree': 0.7, 
 'learning_rate': 0.1, 
'max_depth': 3, 'min_child_weight': 3,
 'n_estimators': 200}
'''
