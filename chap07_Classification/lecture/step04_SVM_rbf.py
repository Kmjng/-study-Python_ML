#########################################
### 선형 vs 비선형 커널 함수 
#########################################

'''
'linear' 커널
선형 커널은 데이터를 선형으로 분리하려고 시도한다. 즉, 데이터를 직선으로 나누는 것을 목표로 한다.
이 커널은 클래스 간의 경계가 선형일 때 유용하다.
'linear' 커널은 비교적 간단하며, 매개변수 튜닝이 적은 편이다.

'rbf' (Radial Basis Function) 커널
'rbf' 커널은 비선형 데이터를 다루는 데 유용하다. 비선형 데이터는 선형 경계로는 분리하기 어려울 때 사용된다.
이 커널은 데이터를 고차원 특징 공간으로 매핑하여 선형 분리 가능한 상태로 변환한다.
'rbf' 커널은 SVM에서 가장 일반적으로 사용되는 커널 중 하나이며, C와 gamma에 민감하며, 
매개변수 튜닝이 중요하다. C는 마진 오류에 대한 패널티를 조절하고, gamma는 커널의 모양을 제어한다.
'''

from sklearn.svm import SVC # svm model 
from sklearn.datasets import make_circles # 중첩된 원형 dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. 데이터셋 로드 : 원형 데이터셋 
X, y = make_circles(noise=0.05, n_samples=200, random_state=123)

X.shape # (200, 2)
y.shape # (200,)
y #  array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0,...])

plt.scatter(X[:, 0], X[:, 1], s=100, c=y,  marker='o') # color = y범주
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. 훈련/검정 데이터셋 생성
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3. 선형 모델 적용 
obj = SVC(C=1.0, kernel='linear')

model = obj.fit(X=X_train, y=y_train)
model.score(X=X_train, y=y_train) # 0.5642857142857143
model.score(X=X_test, y=y_test) # 0.35


# 비선형 모델 적용 
obj2 = SVC(C=1.0, kernel='rbf')

model2 = obj2.fit(X=X_train, y=y_train)

model2.score(X=X_train, y=y_train) # 0.9714285714285714
model2.score(X=X_test, y=y_test) # 0.9833333333333333
















