'''
계층적 군집분석(Hierarchical Clustering) 
 - 상향식(Bottom-up)으로 계층적 군집 형성 
 - 유클리드안 거리계산식 이용 
 - 숫자형 변수만 사용

ppt_12p - sample data 
'''

from scipy.cluster.hierarchy import linkage, dendrogram # 군집분석 도구 
import matplotlib.pyplot as plt # 시각화 
from sklearn.datasets import load_iris # dataset
import pandas as pd # DataFrame

###############sample data로 실습 
## ppt 12p 
df = pd.DataFrame({'x':[1,2,2,4,5], 
                   'y':[1,1,4,3,4]})
df
'''
   x  y
0  1  1  # point 0 
1  2  1  # point 1
2  2  4  # point 2
3  4  3  
4  5  4
'''
# 계층적 군집 모형 
sample_model = linkage(df, method ='single', metric = 'euclidean')
'''
 p          q             거리       노드 수
[0.        , 1.        , 1.        , 2.        ], # point 5 이 된다.
[3.        , 4.        , 1.41421356, 2.        ], # point 6
[2.        , 6.        , 2.23606798, 3.        ], # point 7 
[5.        , 7.        , 2.82842712, 5.        ]]

0 ~ 4까지 기존 point 
5 ~ 7 은 최초 군집 이후 point  
'''
# 모델 객체 기반 덴드로그램 시각화 
plt.figure(figsize = (25,10))
dendrogram(sample_model)
plt.show()
# x축은 각 point, y축은 군집 거리  

###################
# 1. dataset loading
iris = load_iris() # Load the data # 수치형 데이터 

X = iris.data # x변수 
y = iris.target # y변수(target) - 숫자형 : 거리계산 

# X + y 결합 (데이터셋 DataFrame으로 변경)
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y # target 추가 


# 2. 계층적 군집분석 
clusters = linkage(iris_df, method='single')
# clusters ; 모델(model) 객체 이름
'''
군집화 방식 : ppt.10 ~ 11 참고 
method = 'single' : 단순연결(default)
method = 'complete' : 완전연결 
method = 'average' : 평균연결
method = 'centroid' : 두 중심점의 거리 

데이터가 150개라면 
1번째 & (2~150)번째 거리 계산 
2번째 & (3~150)번째 거리 계산 
...

'''
print(clusters)
clusters.shape # (149, 4) ★★★
X.shape # (150, 4)
y.shape # (150,)



# 3. 덴드로그램(dendrogram) 시각화 : 군집수 사용자가 결정 
plt.figure(figsize = (25, 10))
dendrogram(clusters)
plt.show()

# 4. 군집(cluster) 자르기 : fcluster (ppt.17 ~ 18 참고)
from scipy.cluster.hierarchy import fcluster # 군집 자르기 도구 
import numpy as np # 클러스터 빈도수 

cut_cluster = fcluster(clusters, t=3, criterion='maxclust') 
# criterion='maxclust': 최대 클러스터 갯수 t=3 개로 구분하겠다.
# criterion='distance': 군집 거리 t=3 기준으로 구분하겠다. 

cut_cluster # 1~3으로 구성됨 # 각 데이터가 어느 군집인지 알 수 있음
len(cut_cluster) # 150

# 군집(cluster) 빈도수 
unique, counts = np.unique(cut_cluster, return_counts=True)
print(unique, counts)
# [1 2 3] [50 50 50] 

# 5. 군집화 데이터 DF 칼럼으로 추가 ★★
iris_df['cluster'] = cut_cluster
iris_df
'''
     sepal length (cm)  sepal width (cm)  ...  species  cluster
0                  5.1               3.5  ...        0        1
1                  4.9               3.0  ...        0        1
2                  4.7               3.2  ...        0        1
3                  4.6               3.1  ...        0        1
4                  5.0               3.6  ...        0        1
..                 ...               ...  ...      ...      ...
145                6.7               3.0  ...        2        3
146                6.3               2.5  ...        2        3
147                6.5               3.0  ...        2        3
148                6.2               3.4  ...        2        3
149                5.9               3.0  ...        2        3

[150 rows x 6 columns]
'''
# 6. 계층적군집분석 시각화 
plt.scatter(iris_df['sepal length (cm)'], iris_df['petal length (cm)'],
            c=iris_df['cluster']) # (x, y, color )
            # c= 데이터 포인트의 색상 결정
plt.show()

##################################
# 7. 각 군집별 특성 분석 
# 사용자가 해주어야 한다. ★★★ 

# DataFrame 그룹핑 
group = iris_df.groupby('cluster')
group.size() 
'''
cluster
1    50
2    50
3    50
'''

group.mean().T
'''
cluster                1      2      3
sepal length (cm)  5.006  5.936  6.588
sepal width (cm)   3.428  2.770  2.974
petal length (cm)  1.462  4.260  5.552
petal width (cm)   0.246  1.326  2.026
species            0.000  1.000  2.000

cluster 1 : 꽃받침(sepal) 길/넓이 , 꽃잎(petal) 길/넓이 상대적 작음 
cluster 2 : 꽃받침(sepal) 길/넓이 , 꽃잎(petal) 길/넓이 중간 
cluster 3 : 꽃받침(sepal) 길/넓이 , 꽃잎(petal) 길/넓이 상대적 큼 
'''

