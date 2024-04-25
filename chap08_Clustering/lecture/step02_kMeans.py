# -*- coding: utf-8 -*-
"""
kMeans 알고리즘 
 - 확인적 군집분석 
 - 군집수 k를 알고 있는 분석방법 
"""

import pandas as pd # DataFrame 
from sklearn.cluster import KMeans # model 
import matplotlib.pyplot as plt # 군집결과 시각화 
import numpy as np # array 


# 1. text file -> dataset 생성 
file = open('C:/ITWILL/4_Python_ML/data/testSet.txt')
lines = file.readlines() # list 반환 

print(lines) # list 
'''
1.658985	4.285136
-3.453687	3.424321
4.838138	-1.151539
-5.379713	-3.362104
0.972564	2.924086...

x축과 y축 (str형으로 되어 있음) => 데이터 전처리 필요
'''

dataset = [] # 2차원(80x2)
for line in lines : 
    cols = line.split('\t') # tab키 기준 분리 
            # 1.658985\t4.285136
    
    rows = [] # 1줄 저장을 위함
    for col in cols : # 칼럼 단위 추가 
        rows.append(float(col)) # 문자열 -> 실수형 변환  
                          # [1.658985, 4.285136]
    dataset.append(rows)  # [[1.658985, 4.285136],....]
        
print(dataset) # 중첩 list   

# list -> numpy(array) 
dataset_arr = np.array(dataset)

dataset_arr.shape # (80, 2)
print(dataset_arr) # 자료 준비 완료


# 2. numpy -> DataFrame(column 지정)
data_df = pd.DataFrame(dataset_arr, columns=['x', 'y'])
data_df.info()
'''
RangeIndex: 80 entries, 0 to 79
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   x       80 non-null     float64
 1   y       80 non-null     float64
'''
# 이거 왜안되지 
data_df.plot(kind = 'scatter', x = data_df.x, y = data_df.y)

plt.scatter(data_df.x, data_df.y)


# 3. KMeans model 생성 
obj = KMeans(n_clusters=4, max_iter=300, algorithm='auto')

help(obj)
'''
KMeans model Hyper parameter 
algorithm =  {"lloyd", "elkan", "auto", "full"}, default="lloyd"
'''


model = obj.fit(data_df) # 학습 수행 
dir(model)
'''
cluster_centers_ : 각 클러스터의 중심점 반환 
labels_ : 현재 data의 예측된 cluster 
predict() : new data의 예측된 cluster 
'''



# 예측치 생성 
pred = model.predict(data_df) # test set 
print(pred) # 0 ~ 3 의 4개의 군집으로 구성됨 
'''
[0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0
 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3 2 1 0 3
 2 1 0 3 2 1]
'''
model.labels_.shape # (150,) # labels_ 메소드로도 예측 군집 반환받을 수 있음 
pred.shape # (150,)
'''
model.predict(테스트 셋) 
 >> predict 메서드는 학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행할 때 사용
labels_
 >> 학습 데이터에 대한 결과를 분석하는 데 사용
'''

# 군집 중앙값 재설정 ★★★
centers = model.cluster_centers_
print(centers)
'''
       x            y
1[[ 2.6265299   3.10868015]
2 [-3.38237045 -2.9473363 ]
3 [ 2.80293085 -2.7315146 ]
4 [-2.46154315  2.78737555]]
'''

# 예측된 군집을 데이터프레임에 추가 
data_df['predict']=pred

data_df
'''
           x         y  predict
0   1.658985  4.285136        0
1  -3.453687  3.424321        3
..       ...       ...      ...
78  4.479332 -1.764772        2
79 -4.905566 -2.911070        1

[80 rows x 3 columns]
'''

# clusters 시각화 : 예측 결과 확인 

# 산점도 
plt.scatter(x=data_df['x'], y=data_df['y'], 
            c=data_df['predict'])

# 중앙값 추가 
plt.scatter(x=centers[:,0], y=centers[:,1], 
            c='r', marker='D')
plt.show()

# 그룹별 평균 
group = data_df.groupby('predict')

group.size() 
'''
predict
0    20
1    20
2    20
3    20
'''
group.mean()
'''
                x         y
predict                    
0        2.626530  3.108680
1       -3.382370 -2.947336
2        2.802931 -2.731515
3       -2.461543  2.787376
'''