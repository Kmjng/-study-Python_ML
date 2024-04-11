'''
시계열 데이터 시각화 

인덱스를 날짜형(Date)로 지정하면 (set_index('칼럼명'))
범위 선택할 때 다음과 같이 사용 가능하다.
DF명.loc['YYYY'] 
DF명.loc['YYYY-MM']
DF명.loc['YYYY-MM-DD']  
'''

import pandas as pd
import matplotlib.pyplot as plt


# 1. 날짜형식 수정(다국어)
path = r'C:/ITWILL/4_Python_ML/data'
cospi = pd.read_csv(path + "/cospi.csv")


# object -> Date형 변환 
cospi['Date'] = pd.to_datetime(cospi['Date'])
print(cospi.info())

# 2016-01 ~ 2016.02 
new_cospi = cospi[(cospi.Date >= '2016-01') & (cospi.Date <'2016-02')]
new_cospi

# 2. 시계열 데이터/시각화

# 1개 칼럼 추세그래프 
cospi['High'].plot(title = "Trend line of High column")
plt.show()

# 2개 칼럼(중첩list) 추세그래프
cospi[['High', 'Low']].plot(color = ['r', 'b'],
        title = "Trend line of High and Low column")
plt.show() 


# DataFrame의 index 수정 : Date 칼럼 이용  
# set_index() 
new_cospi = cospi.set_index('Date')
print(new_cospi.info())
print(new_cospi.head())

# 날짜형 색인 
new_cospi.index #  DatetimeIndex(['2016-02-26', '2016-02-25',
print(new_cospi.loc['2016']) # 년도 선택 
print(new_cospi.loc['2016-02']) # 월 선택 
print(new_cospi.loc['2016-01':'2016-02']) # 범위 선택 # 오류
# 정렬이 필요하다. 
new_cospi = new_cospi.sort_index() 

'''
DataFrame에서의 정렬 ★★★
 sort_index() : 색인정렬
 sort_values(by ='칼럼명') : 특정 칼럼으로 정렬
'''
new_cospi.High
'''
Date
2015-03-02    1423000
2015-03-03    1437000

2016-02-25    1187000
2016-02-26    1187000
Name: High, Length: 247, dtype: int64
'''
# 2016년도 주가 추세선 시각화 
new_cospi_HL = new_cospi[['High', 'Low']]
new_cospi_HL.loc['2016'].plot(title = "Trend line of 2016 year")
plt.show()

new_cospi_HL.loc['2016-02'].plot(title = "Trend line of 2016 year")
plt.show()


# 3. 이동평균(평활) : 지정한 날짜 단위 평균계산 -> 추세그래프 스무딩  

# 5일 단위 평균계산 : 평균계산 후 5일 시작점 이동 
# rolling() 
roll_mean5 = pd.Series.rolling(new_cospi.High,
                               window=5, center=False).mean()
print(roll_mean5)
'''
Date
2015-03-02          NaN
2015-03-03          NaN
2015-03-04          NaN
2015-03-05          NaN
2015-03-06    1438400.0
   
2016-02-22    1194000.0
'''

# 10일 단위 평균계산 
roll_mean10 = pd.Series.rolling(new_cospi.High,
                               window=10, center=False).mean()
print(roll_mean10)
'''
Date
2015-03-02          NaN
2015-03-03          NaN
2015-03-04          NaN
2015-03-05          NaN
2015-03-06          NaN
   
2016-02-22    1174800.0
2016-02-23    1177600.0
'''
# 1) High 칼럼 시각화 
new_cospi['High'].plot(color = 'blue', label = 'High column')


# 2) rolling mean 시각화 : subplot 이용 - 격자 1개  
fig = plt.figure(figsize=(12,4))
chart = fig.add_subplot()
chart.plot(new_cospi['High'], color = 'blue', label = 'High column')
chart.plot(roll_mean5, color='red',label='5 day rolling mean')
chart.plot(roll_mean10, color = 'black', label='10 day rolling mean')
plt.legend(loc='best')
plt.show()










