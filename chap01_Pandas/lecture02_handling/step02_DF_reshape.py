# -*- coding: utf-8 -*-
"""
step02_DF_reshape.py

- DataFrame 모양 변경 
"""

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

buy = pd.read_csv(path + '/buy_data.csv')

print(buy.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22 entries, 0 to 21
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Date         22 non-null     int64
 1   Customer_ID  22 non-null     int64
 2   Buy          22 non-null     int64
'''


# 1. 2차원 -> 1차원 구조 변경
# stack() 사용
# 데이터프레임.stack() 
buy_long = buy.stack() 
buy_long.shape # >> (66,)
buy_long
'''
0   Date           20150101
    Customer_ID           1
    Buy                   3
1   Date           20150101
    Customer_ID           2
  
20  Customer_ID           1
    Buy                   9
21  Date           20150107
    Customer_ID           5
    Buy                   7
Length: 66, dtype: int64
'''



# 2. 1차원 -> 2차원 구조 변경 
buy_wide = buy_long.unstack()

# 3. 전치행렬 
buy_tran = buy.T
buy_tran.shape # (3,22)

# 4. 중복 행 제거 
buy2 = buy.drop_duplicates() # 중복 행 제거
buy2.shape # (20, 3)

buy2
'''        Date  Customer_ID  Buy
0   20150101            1    3
1   20150101            2    4
2   20150102            1    2

...
19  20150106            3    6
20  20150107            1    9
21  20150107            5    7
'''
# 5. 특정 칼럼을 index 로 지정 (행 이름)
# set_index('칼럼명')
new_buy = buy.set_index('Date') # 구매날짜 
new_buy.shape # (22,2)
new_buy
'''          Customer_ID  Buy
Date                      
20150101            1    3
20150101            2    4
20150102            1    2
...
20150106            3    6
20150107            1    9
20150107            5    7
'''
# 날짜 검색 
new_buy.loc[20150101] # 명칭색인 
#new_buy.iloc[20150101] # 오류 : out-of-bounds : 색인 범위 초과 
# 색인 자료형은 int형이지만 동일한 구매날짜를 명칭으로 지정한다.  

# [추가] 주가 dataset 적용 
stock = pd.read_csv(path+'/stock_px.csv')
stock.info()
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2214 entries, 0 to 2213
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  2214 non-null   object  => to_datetime()메소드로 형 변환하기
 1   AAPL        2214 non-null   float64
 2   MSFT        2214 non-null   float64
 3   XOM         2214 non-null   float64
 4   SPX         2214 non-null   float64
dtypes: float64(4), object(1)
memory usage: 86.6+ KB
'''
# 칼럼명 수정 #Unnamed 칼럼명 설정 
stock.columns =['Date','AAPL','MSFT','XOM','SPX'] 
# object => date형 변환 
stock['Date'] =pd.to_datetime(stock.Date)
stock.info() #  0   Date    2214 non-null   datetime64[ns]

new_stock = stock.set_index('Date')
new_stock
'''              AAPL   MSFT    XOM      SPX
Date                                     
2003-01-02    7.40  21.11  29.22   909.03
2003-01-03    7.45  21.14 
...
'''

app_ms = new_stock[['AAPL','MSFT']]
app_ms.plot()


app_ms.loc['2011'].plot() # 행 이름이 2011
app_ms.loc['2011-03':'2011-07'].plot() # 행 범위
