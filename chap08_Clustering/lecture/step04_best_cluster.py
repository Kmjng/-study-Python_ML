'''
Best Cluster 찾는 방법 
'''

from sklearn.cluster import KMeans # model 
from sklearn.datasets import load_iris # dataset 
import matplotlib.pyplot as plt # 시각화 

# 1. dataset load 
X, y = load_iris(return_X_y=True)
print(X.shape) # (150, 4)
print(X)


# 2. Best Cluster 
'''
엘보우 방법은 K-means 알고리즘을 실행할 때, 클러스터 개수(K)를 점차 증가시키면서 
클러스터링을 수행하고, 이에 따른 SSE(Sum of Squared Errors) 값을 계산하여 
그래프로 나타내어 최적의 K값을 선택하는 방법

SSE 값 계산 : inertia_ 메소드

inertia value (y값) 가 작을 수록 '응집도'가 좋다.

'''
size = range(1, 11) # k값 범위
inertia = [] # 응집도 # 모든 point ~ 중심point 간의 거리제곱의 합 



for k in size :  # k = 1 ~ 10 
    obj = KMeans(n_clusters = k) 
    model = obj.fit(X)
    inertia.append(model.inertia_) 

print(inertia)


# 3. best cluster 찾기 
plt.plot(size, inertia, '-o')
plt.xticks(size)
plt.show()
