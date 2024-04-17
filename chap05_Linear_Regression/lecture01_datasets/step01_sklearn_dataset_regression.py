'''
회귀분석용 sklearn dataset 정리 

sklearn 제공 데이터셋 : DataFrame형태 아님 (데이터프레임으로 쓰려면 바꿔줄 것)
    type : <class 'sklearn.utils._bunch.Bunch'>
데이터셋 객체의 메소드
iris.DESCR
iris.feature_names # 피쳐이름(칼럼명)
iris.target_names # 피쳐이름(칼럼명)
iris_X = iris.data # x변수 만들기
iris_y = iris.target # y변수 만들기 

# DataFrame 변환  
import pandas as pd
iris_df = pd.DataFrame(x변수데이터셋, columns=iris.feature_names)
iris_df['species'] = iris_y  # y변수 추가 
'''
from sklearn import datasets # dataset 제공 library

# sklearn 모듈 확인 .__all__ 
import sklearn
print(sklearn.__all__)
'''
★★★
cluster  : 군집모델
datasets : 제공데이터셋
ensemble : 앙상블모델
linear_model
metrics  : 모델 평가도구 
model_selection : train/test set 나누기
neural_network : 신경망 모델 
preprocessing 
svm
tree
'''
######################################
# 선형회귀분석에 적합한 데이터셋
######################################

# 1. 붓꽃(iris) : 회귀와 분류 모두 사용 
'''
붓꽃(iris) 데이터
- 붓꽃 데이터는 통계학자 피셔(R.A Fisher)의 붓꽃의 분류 연구에 기반한 데이터

• 타겟 변수 : y변수
세가지 붓꽃 종(species) : setosa, versicolor, virginica

•특징 변수(4) : x변수
꽃받침 길이(Sepal Length)
꽃받침 폭(Sepal Width)
꽃잎 길이(Petal Length)
꽃잎 폭(Petal Width)
'''
iris = datasets.load_iris() # dataset load 
print(type(iris))  # <class 'sklearn.utils._bunch.Bunch'> ★★★★

# dataset 설명제공 : 변수특징, 요약통계 
print(iris.DESCR) 
'''
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
'''

# X, y변수 선택 
iris_X = iris.data # x변수 
iris_y = iris.target # y변수

iris_X
'''
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
'''
iris_y # 0,1,2으로 구성되어 있는 배열


# 객체형과 모양확인 
print(type(iris_X)) # >> <class 'numpy.ndarray'>
print(type(iris_y)) # >> <class 'numpy.ndarray'>

print(iris_X.shape) # (150, 4) : 2차원
print(iris_y.shape) # (150,)   : 1차원 


# X변수명과 y변수 범주명 
# ★★★ 배열은 feature name(칼럼명)이 없기 때문에 필요하다
# iris 데이터셋(데이터프레임)의 feature name을 가져온다. 
print(iris.feature_names)# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']


# DataFrame 변환  
import pandas as pd
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)

# y변수 추가 
iris_df['species'] = iris.target 
iris_df.head()
iris_df.info() 


# 차트 분석 : 각 특징별 타겟변수의 분포현황  
import seaborn as sn
import matplotlib.pyplot as plt

# 변수 간 산점도 : hue = 집단변수 : 집단별 색상 제공 
sn.pairplot(iris_df ,hue="species")
plt.show() 


# 2. 당뇨병 데이터셋
'''
- 442명의 당뇨병 환자를 대상으로한 검사 결과를 나타내는 데이터

•타겟 변수 : y변수
1년 뒤 측정한 당료병 진행상태 정량적화 자료(연속형)

•특징 변수(10: 모두 정규화된 값) : x변수
age : 나이 (세)
sex : 성별 
bmi : 비만도지수
bp : 평균혈압(Average blood pressure)
S1 ~ S6: 기타 당료병에 영향을 미치는 요인들 
'''

diabetes = datasets.load_diabetes() # dataset load 
print(diabetes.DESCR) # 컬럼 설명, url
'''
:Target: Column 11 -> 1년기준으로 질병 진행상태를 정량적(연속형)으로 측정 
:Attribute Information: Age ~ S6
'''    

print(diabetes.feature_names) # X변수명 
#print(diabetes.target_names) # None : 연속형 변수 이름 없음 

# X, y변수 동시 선택 
X, y = datasets.load_diabetes(return_X_y=True)

print(X.shape) # (442, 10) 
print(y.shape) # (442,) 



# 3. california 주택가격 
'''
•타겟 변수 : y변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) : x변수
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)

# X변수 -> DataFrame 변환 
cal_df = pd.DataFrame(california.data, columns=california.feature_names)
# y변수 추가 
cal_df["MEDV"] = california.target
cal_df.tail()
cal_df.info() 

