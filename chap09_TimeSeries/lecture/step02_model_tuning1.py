"""
모델 튜닝1(model tuning1) : 휴일(holiday) 파라미터 추가
"""

from prophet import Prophet

import pandas as pd # dataset
from sklearn.metrics import r2_score # 평가 


### 1. dataset load 

path = r'C:\ITWILL\4_Python_ML\data\Bike-Sharing-Dataset'
data = pd.read_csv(path + '/day.csv')
data.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 731 entries, 0 to 730
Data columns (total 16 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 1   dteday      731 non-null    object : 날짜 
 2   season      731 non-null    int64  : 계절 
 3   yr          731 non-null    int64  : 연도 
 4   mnth        731 non-null    int64  : 월 
 5   holiday     731 non-null    int64  : 휴일 
     :
 13  casual      731 non-null    int64   : 비가입자 이용수 
 14  registered  731 non-null    int64   : 가입자 이용수 
'''
data.head()


# 변수 선택 
new_data = data[['dteday','registered']]

# 날짜/시간 자료형 변환 
new_data['dteday'] = pd.to_datetime(new_data['dteday'])

 # 칼럼명 수정 
new_data.columns = ['ds', 'y']



### 2. train vs test split : 2012-11-01 기준 
train = new_data[(new_data.ds >= '2011-01-01') & (new_data.ds <= '2012-10-31')]
test = new_data[new_data.ds >= '2012-11-01']

train.shape # (670, 2)
test.shape #  (61, 2)


##############################################
## model tuning : 휴일(holiday) 파라미터 추가 
##############################################

# 휴일 자료 만들기 : 휴일 추출 
holidays = data.loc[data['holiday'] == 1, 'dteday']

# 데이터프레임 만들기 
holi_df = pd.DataFrame({'holiday':'holiday', # 가변수 
                            'ds':holidays,   # 휴일 날짜 정보 
                            'lower_window':0,  
                            'upper_window':0})
holi_df 
'''
     holiday         ds  lower_window  upper_window
16   holiday 2011-01-17             0             0
51   holiday 2011-02-21             0             0
104  holiday 2011-04-15             0             0
'''



### 4. model 생성 

# 1) 기본 모델
model = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative')
model.fit(train)  

# 2) 모델 튜닝 
# 성능 개선 
model2 = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays= holi_df,   # 휴일 자료를 추가해 튜닝된 모델 
                seasonality_mode='multiplicative')
model2 = model2.fit(train)   



### 5. 예측용 데이터 생성 & 예측 

# 1) 기본 모델 예측 
future_data = model.make_future_dataframe(periods=61, freq='D')  
future_pred = model.predict(future_data)  

# 2) 모델 튜닝 예측 
future_pred2 = model2.predict(future_data) 


### 6. model 평가 
y_test = test.y

yhat = future_pred.iloc[-61:, -1] # model1 예측치 
yhat2 = future_pred2.iloc[-61:, -1] # model2 예측치 

# 1) R2 점수 
score = r2_score(y_test, yhat)
print(f'model1 - r2 score : {score : .5f}') 

score = r2_score(y_test, yhat2)
print(f'model2 - r2 score : {score : .5f}') 
'''
model1 - r2 score :  0.37451
model2 - r2 score :  0.41457   # 4% 개선됨 
'''

# 2) 시계열 예측 결과 : model1 vs model2
import matplotlib.pyplot as plt 
plt.plot(test.ds, y_test, c='b', label='real data')
plt.plot(test.ds, yhat, c='g', label = 'model1 predicted data')
plt.plot(test.ds, yhat2, c='r', label = 'model2 predicted data')
plt.legend()
plt.xticks(rotation=90)
plt.show()
