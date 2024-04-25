# -*- coding: utf-8 -*-

'''
UCI ML Repository 데이터셋 url
https://archive.ics.uci.edu/ml/datasets.php
'''

### 기본 라이브러리 불러오기
import pandas as pd
pd.set_option('display.max_columns', 100) # 콘솔에서 보여질 최대 칼럼 개수 
import matplotlib.pyplot as plt



### [Step 1] 데이터 준비 : 도매 고객 데이터셋 
'''
 - 도매 유통업체의 고객 관련 데이터셋으로 다양한 제품 범주에 대한 연간 지출액을 포함  
 - 출처: UCI ML Repository
'''
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path)
df.info() # 변수 및 자료형 확인
'''
RangeIndex: 440 entries, 0 to 439
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   Channel           440 non-null    int64 : 유통업체 : Horeca(호텔/레스토랑/카페) 또는 소매(명목)
 1   Region            440 non-null    int64 : 지역 : Lisnon,Porto 또는 기타(명목) - 리스본,포르토(포르투갈)  
 2   Fresh             440 non-null    int64 : 신선함 : 신선 제품에 대한 연간 지출 ★(연속)
 3   Milk              440 non-null    int64 : 우유 : 유제품에 대한 연간 지출 ★(연속)
 4   Grocery           440 non-null    int64 : 식료품 : 식료품에 대한 연간 지출 ★(연속)
 5   Frozen            440 non-null    int64 : 냉동 제품 : 냉동 제품에 대한 연간 지출 ★(연속)
 6   Detergents_Paper  440 non-null    int64 : 세제-종이 : 세제 및 종이 제품에 대한 연간 지출 ★(연속)
 7   Delicassen        440 non-null    int64 : 델리카슨 : 델리카트슨(수입식품) 제품 ★(연속)
'''


### [Step 2] 데이터 탐색

# 데이터 살펴보기
print(df.head())  
'''
   Channel  Region  Fresh  Milk  Grocery  Frozen  Detergents_Paper  Delicassen
0        2       3  12669  9656     7561     214              2674        1338
1        2       3   7057  9810     9568    1762              3293        1776
2        2       3   6353  8808     7684    2405              3516        7844
3        1       3  13265  1196     4221    6404               507        1788
4        2       3  22615  5410     7198    3915              1777        5185
'''
# 명목형 변수 
df.Channel.value_counts() # 유통업체
'''
1    298 # 1) Horeca
2    142 # 2) 소매 
'''

df.Region.value_counts() # 유통 지역
'''
3    316 # 3) 기타 
1     77 # 1) Lisnon
2     47 # 2) Porto 
'''

# 연속형 변수 
df.describe() # 나머지 연속형 변수(각 변수 척도 다름) 


### [Step 3] 데이터 전처리

# 분석에 사용할 변수 선택
X = df.copy()

# 설명변수 데이터 정규화
from sklearn.preprocessing import StandardScaler # 표준화 
X = StandardScaler().fit_transform(X)


### [Step 4] k-means 군집 모형 - sklearn 사용

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn.cluster import KMeans

# 모형 객체 생성 
kmeans = KMeans(n_clusters=5, n_init=10, max_iter=300, 
                random_state=45) # cluster 5개 

'''
Parameters
----------
n_clusters : int, default=8
n_init : int, default=10 - centroid seeds
max_iter : int, default=300
random_state : 모델 시드값 
'''
        
# 모형 학습
kmeans.fit(X)  # KMeans(n_clusters=5) 

# 군집 예측 (분류된 군집 반환)
cluster_labels = kmeans.labels_ # 예측된 레이블(Cluster 번호) ★★★
print(cluster_labels)



# 데이터프레임에 예측된 레이블 추가
df['Cluster'] = cluster_labels
print(df.head())   

# 상관관계 분석 
r = df.corr() # 상관계수가 높은 변수 확인 
r.T
'''
                   Channel    Region     Fresh      Milk   Grocery    Frozen  \
Channel           1.000000  0.062028 -0.169172  0.460720  0.608792 -0.202046   
Region            0.062028  1.000000  0.055287  0.032288  0.007696 -0.021044   
Fresh            -0.169172  0.055287  1.000000  0.100510 -0.011854  0.345881   
Milk              0.460720  0.032288  0.100510  1.000000  0.728335  0.123994   
Grocery           0.608792  0.007696 -0.011854  0.728335  1.000000 -0.040193   
Frozen           -0.202046 -0.021044  0.345881  0.123994 -0.040193  1.000000   
Detergents_Paper  0.636026 -0.001483 -0.101953  0.661816  0.924641 -0.131525   
Delicassen        0.056011  0.045212  0.244690  0.406368  0.205497  0.390947   
Cluster          -0.288395  0.774399  0.088331 -0.165918 -0.218906  0.075043   

                  Detergents_Paper  Delicassen   Cluster  
Channel                   0.636026    0.056011 -0.288395  
Region                   -0.001483    0.045212  0.774399  
Fresh                    -0.101953    0.244690  0.088331  
Milk                      0.661816    0.406368 -0.165918  
Grocery                   0.924641    0.205497 -0.218906  
Frozen                   -0.131525    0.390947  0.075043  
Detergents_Paper          1.000000    0.069291 -0.224032  
Delicassen                0.069291    1.000000 -0.000279  
Cluster                  -0.224032   -0.000279  1.000000  
'''
 
# 그래프로 표현 - 시각화
df.plot(kind='scatter', x='Grocery', y='Detergents_Paper', c='Cluster', 
        cmap='Set1', colorbar=True, figsize=(15, 10))
plt.show()  

# 각 클러스터 빈도수 : 빈도수가 적은 클러스터 제거 
print(df.Cluster.value_counts())
'''
4    212
2    125
0     91
1     11  -> 클러스터 1 제거 
3      1  -> 클러스터 3 제거 
'''

# 새로운 dataset 만들기 : 1, 3번 클러스터 제거 예 
new_df = df[~((df['Cluster'] == 3) | (df['Cluster'] == 1))]

new_df.shape #  (428, 9)


# Grocery vs Detergents_Paper 산점도 

new_df.plot(kind='scatter', x='Grocery', y='Detergents_Paper', c='Cluster', 
        cmap='Set1', colorbar=True, figsize=(15, 10))
plt.show()  

### [Step 5] 각 cluster별 특성 분석
group = new_df.groupby('Cluster')
group.mean().T

'''
Cluster                      0          2             4
Channel               1.054945      2.000      1.004717
Region                1.307692      2.672      2.995283
Fresh             12183.945055   7877.640  13980.273585
Milk               3254.714286   8913.512   3360.995283
Grocery            4130.923077  14212.624   3860.905660
Frozen             3458.252747   1339.280   3760.872642
Detergents_Paper    860.263736   6149.592    790.311321
Delicassen         1149.934066   1537.168   1321.976415
'''

Cluster0 = new_df[new_df.Cluster == 0]
Cluster2 = new_df[new_df.Cluster == 2]
Cluster4 = new_df[new_df.Cluster == 4]

Cluster0.Channel.value_counts()
'''
Channel
1    86
2     5
'''
