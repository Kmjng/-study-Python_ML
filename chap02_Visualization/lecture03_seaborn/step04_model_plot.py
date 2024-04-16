# -*- coding: utf-8 -*-
"""
 - 분석모델 관련 시각화
 seaborn의 lineplot(x='시간축', y='통계량', data=데이터셋객체, hue='집단변수') 
 : 시계열
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn

# seaborn 한글과 음수부호, 스타일 지원 
sn.set(font="Malgun Gothic", 
            rc={"axes.unicode_minus":False}, style="darkgrid")


# dataset 로드 
flights = sn.load_dataset('flights')
flights.info()
'''
 0   year        144 non-null    int64    -> 년도 : x
 1   month       144 non-null    category -> 월  : 집단변수(hue)
 2   passengers  144 non-null    int64    -> 탑승객 : y
'''

iris = sn.load_dataset('iris')
iris.info()

# 1. 오차대역폭을 갖는 시계열 : x:시간축, y:통계량 
sn.lineplot(x = 'year', y = 'passengers', data = flights)
plt.show()

# hue 추가 
sn.lineplot(x = 'year', y = 'passengers', hue='month',
            data = flights)
plt.show()
 

# 2. 선형회귀모델 : 산점도 + 회귀선 
sn.regplot(x = 'sepal_length', y = 'petal_length',  data = iris)  


# 3. heatmap : 분류분석 평가 
y_true = pd.Series([1,0,1,1,0]) # 정답 
y_pred = pd.Series([1,0,0,1,0]) # 예측치 

# 1) 교차분할표(혼동 행렬) ★★★
tab = pd.crosstab(y_true, y_pred, 
            rownames=['관측치'], colnames=['예측치'])
tab
'''
예측치 0  1
관측치      
0      2  0
1      1  2
'''

# 2) heatmap
sn.heatmap(data=tab, annot = True) # annot = True : box에 빈도수 
plt.show()

