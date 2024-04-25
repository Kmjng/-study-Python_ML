# -*- coding: utf-8 -*-
"""
<준비물> 
prophet 패키지 설치 
pip install prophet


단순(simple)시계열모델 : x변수 1개 -> y변수 1개
   ex) 과거 목적변수 -> 미래 목적변수 예측 
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
 0   instant     731 non-null    int64  : 일련번호
 1   dteday      731 non-null    object : 날짜    => x축 (시간축) 
 2   season      731 non-null    int64  : 계절 
 3   yr          731 non-null    int64  : 연도 
 4   mnth        731 non-null    int64  : 월 
 5   holiday     731 non-null    int64  : 휴일 
 6   weekday     731 non-null    int64  : 요일 
 7   workingday  731 non-null    int64  : 근무일 
 8   weathersit  731 non-null    int64  : 날씨 
 9   temp        731 non-null    float64 : 온도
 10  atemp       731 non-null    float64 : 체감온도
 11  hum         731 non-null    float64 : 습도
 12  windspeed   731 non-null    float64 : 풍속
 13  casual      731 non-null    int64   : 비가입자이용수 
 14  registered  731 non-null    int64   : 가입자이용수  => y축 (통계량) 
 15  cnt         731 non-null    int64   : 전체사용자이용수 
'''
data.head()


# 변수 선택 : 날짜, 가입자이용수
df = data[['dteday','registered']]
df.shape #  (731, 2)

# 날짜/시간 자료형 변환 
df['dteday'] = pd.to_datetime(df['dteday'])
 
# 칼럼명 수정 
df.columns = ['ds', 'y']


### 2. 시계열자료 추세 & 계절성 확인 

### 시각화 ###
import matplotlib.pyplot as plt 
fig = plt.figure(figsize = (12, 5)) 
chart = fig.add_subplot()  

chart.plot(df.ds, df.y, marker='.', label='time series data')
plt.xlabel('year-month')
plt.ylabel('user number')
plt.legend(loc='best')
plt.show()


### 3. train & test split  
# 훈련셋
train = df[(df.ds >= '2011-01-01') & (df.ds <= '2012-10-31')]

# 예측치 (정답을 준비함)
test = df[df.ds >= '2012-11-01'] 


### 4. 시계열모델 생성 
model = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative')
model.fit(train)
'''
yearly_seasonality : 연단위 주기 (봄, 여름, 가을, 겨울)
weekly_seasonality : 주단위 주기
daily_seasonality : 일단위 주기
seasonality_mode : 가법(additive) 또는 승법(multiplicative) 모델 선택 
'''


### 5. 예측용 데이터 생성 & 예측 
future_date = model.make_future_dataframe(periods=61, freq='D') # 예측 미래시점  
future_pred = model.predict(future_date) # 모델 예측 
# 731 = 670(과거) + 61(미래)

### 6. 시계열모델 평가 
### 시각화 ###

# 1) 요소분해 : 추세, 계절성 
# trend, weekday에 대한 계절성, yearly에 대한 계절성
model.plot_components(future_pred)
plt.show()


# 2) 시계열모델 예측 결과  
fig, ax = plt.subplots(figsize = (12, 5))

model.plot(fcst=future_pred, ax=ax) # ax 적용 
ax.set_title('total user number')
ax.set_xlabel('Date')
ax.set_ylabel('user number')
plt.show()

'''
- 연한 테두리 : 신뢰구간에 해당 
        신뢰구간 안에 point가 들어가면 good , 밖이면 오분류로 판단

'''

# 3) 평가 : 예측치 vs 관측치  
y_pred = future_pred.iloc[-61:, -1]
y_test = test.y

score = r2_score(y_test, y_pred)
score # 0.37451445994051946

# 4) 시계열자료 vs 모델 예측 
plt.plot(test.ds, y_test, c='b', label='real data')
plt.plot(test.ds, y_pred, c='r', label = 'predicted data')
plt.legend()
plt.xticks(rotation=90)
plt.show()
