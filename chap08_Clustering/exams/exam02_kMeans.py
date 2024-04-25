# -*- coding: utf-8 -*-
"""
문2) 아래와 같은 단계로 kMeans 알고리즘을 적용하여 확인적 군집분석을 수행하시오.

 <조건> 변수 설명 : tot_price : 총구매액, buy_count : 구매횟수, 
                   visit_count : 매장방문횟수, avg_price : 평균구매액

  단계1 : 3개 군집으로 군집화 
  단계2: 원형데이터에 군집 예측치 추가  
  단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)  
  단계4 : 산점도에 군집의 중심점 시각화
  단계5 : 군집별 특성분석 : 우수고객 군집 식별
"""

import pandas as pd
from sklearn.cluster import KMeans # kMeans model
import matplotlib.pyplot as plt

sales = pd.read_csv("C:/ITWILL/4_Python_ML/data/product_sales.csv")
print(sales.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
tot_price      150 non-null float64 -> 총구매금액 
visit_count    150 non-null float64 -> 매장방문수 
buy_count      150 non-null float64 -> 구매횟수 
avg_price      150 non-null float64 -> 평균구매금액 
'''


sales_corr = sales.corr().T.tot_price[1:]
'''
visit_count    0.817954
buy_count     -0.013051
avg_price      0.871754
Name: tot_price, dtype: float64
'''


# 단계1 : 3개 군집으로 군집화
model = KMeans(n_clusters = 3, max_iter = 300, algorithm='auto', random_state=123).fit(sales)


pred = model.predict(sales) # test set 

centers = model.cluster_centers_
print(centers)
'''
[[5.9016129  1.43387097 2.75483871 4.39354839]
 [5.006      0.244      3.284      1.464     ]
 [6.85       2.07105263 3.07105263 5.74210526]]
'''
# 단계2: 원형데이터에 군집 예측치 추가
sales['predict'] =pred


# 단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)

Top_corr= sales_corr.sort_values(ascending = False)
Top_corr = Top_corr.index[0]
Top_corr # avg_price

# 단계4 : 산점도에 군집의 중심점 시각화
plt.scatter(x= sales[Top_corr], y = sales['tot_price'], c = sales['predict'])
plt.scatter(x=centers[:,3], y=centers[:,0], # 인덱스 잘 확인하기 ★★★
            c='r', marker='D')
plt.xlabel('avg_price')
plt.ylabel('tot_price')
plt.show()

# 단계5 : 군집별 특성분석 : 우수고객 군집 식별
group = sales.groupby('predict')
group.mean().T
'''
predict             0      1         2
tot_price    5.901613  5.006  6.850000
visit_count  1.433871  0.244  2.071053
buy_count    2.754839  3.284  3.071053
avg_price    4.393548  1.464  5.742105 
'''