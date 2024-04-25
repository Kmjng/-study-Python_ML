# -*- coding: utf-8 -*-
"""
모델 튜닝2(model tuning2) : 날씨 관련 독립변수 추가

 - 다중(multiply)시계열모델 : x변수 2개 이상(y변수 1개)
"""

from prophet import Prophet # 프로펫 시계열분석 알고리즘 
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
     : 
 5   holiday     731 non-null    int64  : 휴일 
     :
 8   weathersit  731 non-null    int64  : 날씨 
 9   temp        731 non-null    float64 : 온도
 11  hum         731 non-null    float64 : 습도
 12  windspeed   731 non-null    float64 : 풍속
    :
 14  registered  731 non-null    int64   : 가입자 이용수 
'''
data.head()

data.holiday.unique() # [0,1]



# 변수 선택 
new_data = data[['dteday','registered']]
new_data.shape #  (731, 2)

# 날짜/시간 자료형 변환 
new_data['dteday'] = pd.to_datetime(new_data['dteday'])
new_data.info()
'''
 0   dteday  731 non-null    datetime64[ns]
 1   casual  731 non-null    int64 
'''

# 칼럼명 수정 
new_data.columns = ['ds', 'y']



##############################################
## model tuning1 : 휴일(holiday) 파라미터 추가 
##############################################
# 휴일 자료 만들기 : 휴일 추출 
holidays = data.loc[data['holiday'] == 1, 'dteday']

# 데이터프레임 만들기 
holi_df = pd.DataFrame({'holiday':'holiday',
                            'ds':holidays,
                            'lower_window':0,
                            'upper_window':0})


##############################################
## model tuning2 : 날씨 관련 칼럼추가 
##############################################
# 날씨 관련 칼럼 : 날씨, 온도, 습도, 풍속 추가 

# hum, windspeed, temp
add_data = data[['weathersit','temp','windspeed','hum']]
new_data2 = pd.concat(objs=[new_data,add_data], axis = 1)
new_data2
'''
            ds     y  weathersit      temp  windspeed       hum
0   2011-01-01   654           2  0.344167   0.160446  0.805833
1   2011-01-02   670           2  0.363478   0.248539  0.696087
2   2011-01-03  1229           1  0.196364   0.248309  0.437273
3   2011-01-04  1454           1  0.200000   0.160296  0.590435
4   2011-01-05  1518           1  0.226957   0.186900  0.436957
'''


### 3. train vs test split : 2012-11-01 기준 
train = new_data[(new_data.ds >= '2011-01-01') & (new_data.ds <= '2012-10-31')]
test = new_data[new_data.ds >= '2012-11-01']

# new dataset 
train2 = new_data2[(new_data2.ds >= '2011-01-01') & (new_data2.ds <= '2012-10-31')]
test2 = new_data2[new_data2.ds >= '2012-11-01']

train2.shape # (670, 6)
test2.shape #  (61, 6)
# 670개 학습하여 61개 예측 


### 4. model 생성 

# 1) 기본 모델 
model = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative')
model.fit(train)

# 2) 단변량(univariate)시계열모델 : 독립변수 1개 
model2 = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays= holi_df,   # 휴일정보 추가 
                seasonality_mode='multiplicative')
model2 = model2.fit(train2)   

# 3) 다중(multiply)시계열모델 : 독립변수 2개 이상  
model3 = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays= holi_df,
                seasonality_mode='multiplicative')

# model3 에 독립변수 추가 ★★★
model3.add_regressor('weathersit') # 날씨
model3.add_regressor('temp') # 온도 
model3.add_regressor('hum') # 습도 
model3.add_regressor('windspeed') # 풍속

# 모델 학습 : 훈련셋 반영 
model3 = model3.fit(train2)



### 5. 예측용 데이터 생성 & 예측 

# 1) model1 예측 
future_data = model.make_future_dataframe(periods=61, freq='D') # 주기=61일, 단위=매일 
future_pred = model.predict(future_data) # 관측값+예측값 

# 2) model2 예측 
future_data2 = model2.make_future_dataframe(periods=61, freq='D') # 주기=61일, 단위=매일 
future_pred2 = model2.predict(future_data2) # 관측값+예측값 

# 3) model3 예측 
future_data3 = test2[['weathersit','ds','temp','hum','windspeed']] # 독립변수 
future_pred3 = model3.predict(future_data3) # 관측값+예측값 


### 6. model 평가 
y_test = test2.y

yhat = future_pred.iloc[-61:, -1] # model1 예측치 
yhat2 = future_pred2.iloc[-61:, -1] # model2 예측치 
yhat3 = future_pred3.iloc[-61:, -1] # model3 예측치 

score = r2_score(y_test, yhat)
print(f'model1 - r2 score : {score : .5f}') # r2 score :  0.37295

score = r2_score(y_test, yhat2)
print(f'model2 - r2 score : {score : .5f}') # r2 score :  0.41457

score = r2_score(y_test, yhat3)
print(f'model3 - r2 score : {score : .5f}') # r2 score :  0.61938


# 시계열 예측 결과 : model2 vs model3
import matplotlib.pyplot as plt 
plt.plot(test.ds, y_test, c='blue', label='real data')
plt.plot(test.ds, yhat2, c='green', label = 'model2 predicted data')
plt.plot(test2.ds, yhat3, c='red', linestyle='-',label = 'model3 predicted data')
plt.legend()
plt.xticks(rotation=90)
plt.show()


 