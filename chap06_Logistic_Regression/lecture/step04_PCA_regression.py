# -*- coding: utf-8 -*-
"""
step04_PCA_regression.py

주성분 분석(PCA : Principal Component Analysis)
 1. 다중공선성의 진단 :  다중회귀분석모델 문제점 발생  
 2. 차원 축소 : 특징 수를 줄여서 다중공선성 문제 해결 
"""

from sklearn.decomposition import PCA # 주성분 분석 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols # 최소제곱법
import pandas as pd 

  
# 1.iris dataset load      
iris = load_iris()

X = iris.data
y = iris.target
'''
array([[1. , 5.1, 3.5, 1.4, 0.2],
       [1. , 4.9, 3. , 1.4, 0.2],
       [1. , 4.7, 3.2, 1.3, 0.2],
       [1. , 4.6, 3.1, 1.5, 0.2],
       [1. , 5. , 3.6, 1.4, 0.2],
'''       

df = pd.DataFrame(X, columns= ['x1', 'x2', 'x3', 'x4'])
corr = df.corr()
print(corr)
'''
          x1        x2        x3        x4
x1  1.000000 -0.117570  0.871754  0.817941
x2 -0.117570  1.000000 -0.428440 -0.366126
x3  0.871754 -0.428440  1.000000  0.962865
x4  0.817941 -0.366126  0.962865  1.000000
'''
df['y'] = y 
df.columns  # ['x1', 'x2', 'x3', 'x4', 'y']


# 2. 다중선형회귀분석 (Multiple Linear Regression)
# (중요) 귀무가설 : 독립변수 간 상관관계가 없다(독립이다) ★★★
ols_obj = ols(formula='y ~ x1 + x2 + x3 + x4', data = df)
# formula = '종속변수 ~ 독립변수1 + 독립변수2 + ...'
model = ols_obj.fit()
# 회귀분석 결과 제공  
print(model.summary()) 
'''
Model:                            OLS   Adj. R-squared:                  0.928
Method:                 Least Squares   F-statistic:                     484.5
Date:                Thu, 05 May 2022   Prob (F-statistic):           8.46e-83
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.1865      0.205      0.910      0.364      -0.218       0.591
x1            -0.1119      0.058     -1.941      0.054      -0.226       0.002
x2            -0.0401      0.060     -0.671      0.503      -0.158       0.078
x3             0.2286      0.057      4.022      0.000       0.116       0.341
x4             0.6093      0.094      6.450      0.000       0.423       0.796
==============================================================================

- F통계량 : 0.05 이하면 '통계적으로 유의한 모델이다.'
    (probaility: 분포도에서 면적)
- Adj. R-square : 결정계수; '모델의 설명력'
- std-err : 표준오차 (표본평균들의 표준편차) ; 추정치의 정확성 
- t통계량 : t검정 통계량 
- P>|t| : 유의확률 (t검정 통계량에 대한 확률) 유의수준 5% 기준으로 가설 채택 또는 기각
'''


#  3. 다중공선성의 진단
'''
분산팽창요인(VIF, Variance Inflation Factor) : 다중공선성 진단  
통상적으로 10보다 크면 다중공선성이 있다고 판단
''' 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 형식) variance_inflation_factor(exog, exog_idx)
dir(ols_obj)
'''
exog : 독립변수
endog : 종속변수 
'''
exog = ols_obj.exog # 엑소(exog) # iris.data
exog
'''
array([[1. , 5.1, 3.5, 1.4, 0.2],
       [1. , 4.9, 3. , 1.4, 0.2],
       [1. , 4.7, 3.2, 1.3, 0.2],
       [1. , 4.6, 3.1, 1.5, 0.2],
       [1. , 5. , 3.6, 1.4, 0.2],....
'''
# 다중공선성 진단  
for idx in range(1,5) : # 1~4
    print(variance_inflation_factor(exog, idx)) # idx=1~4
'''
< 다중공선성> 
7.072722013939533
2.1008716761242523
31.26149777492164 # 높은 상관성
16.090175419908462 # 높은 상관성
'''

    
# 4. 주성분분석(PCA)

# 1) 주성분분석 모델 생성 
pca = PCA() # random_state=123
X_pca = pca.fit_transform(X) # 독립변수만 fit 
#_transform 의미: '주성분'에 맞는 데이터로 변형 ★★★
print(X_pca)
X_pca.shape # (150,4)
# 2) 고유값이 설명가능한 분산비율(분산량)
var_ratio = pca.explained_variance_ratio_ # 총 분산 합이 85 % 이상이 되도록 선택 
print(var_ratio) 
'''
    PCA1       PCA2        PCA3      PCA4
[0.92461872 0.05306648 0.01710261 0.00521218]
'''
type(var_ratio) # array
var_ratio[:2].sum() # 0.977685206318795


# 3) 스크리 플롯 : 주성분 개수를 선택할 수 있는 그래프(Elbow Point : 완만해지기 이전 선택)
plt.bar(x = range(4), height=var_ratio)
plt.plot(var_ratio, color='r', linestyle='--', marker='o') ## 선 그래프 출력
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.xticks(range(4), labels = range(1,5))
plt.show()


# 4) 주성분 결정 : 분산비율(분산량) 95%에 해당하는 지점
print(X_pca[:, :2]) # 주성분분석 2개 차원 선정  
X_new = X_pca[:, :2]

X_new.shape # (150, 2) 
print(X_new)

# 5. 주성분분석 결과를 회귀분석과 분류분석의 독립변수 이용 
# ★★★★★★★

from sklearn.linear_model import LinearRegression # 선형회귀모델  
from sklearn.linear_model import LogisticRegression # 로지스틱회귀모델  

##################################
# LinearRegression : X vs X_new
##################################

# 원형 자료 
lr_model1 = LinearRegression().fit(X = X, y = y)
lr_model1.score(X = X, y = y) # r2 score 
# 0.9303939218549564

# 주성분 자료 
lr_model2 = LinearRegression().fit(X = X_new, y = y)
lr_model2.score(X = X_new, y = y) # r2 score
# 0.9087681620170027

##################################
# LogisticRegression : X vs X_new
##################################

# 원형 자료 
lr_model1 = LogisticRegression().fit(X = X, y = y)
lr_model1.score(X = X, y = y) # accuracy
# 0.9733333333333334

# 주성분 자료 
lr_model2 = LogisticRegression().fit(X = X_new, y = y)
lr_model2.score(X = X_new, y = y) # accuracy
# 0.9666666666666667
