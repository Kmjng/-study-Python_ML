# -*- coding: utf-8 -*-
"""
scipy 패키지 이용 
 1. 단순선형회귀분석 
 2. 다중선형회귀분석 
"""

from scipy import stats
import pandas as pd

#귀무가설(H0) : iq는 score에 영향을 미치지 않는다.
#대립가설(H1) : iq는 score에 영향을 미친다.

score_iq = pd.read_csv('c:/itwill/4_python_ml/data/score_iq.csv')
score_iq.info()

# 1. 단순선형회귀분석 
'''
x -> y
'''

# 1) 변수 생성 
x = score_iq['iq'] # 독립변수 
y = score_iq['score'] # 종속변수 

# 2) model 생성 
model = stats.linregress(x, y)
print(model)
'''
LinregressResult(
    slope=0.6514309527270075, : x 기울기 
    intercept=-2.8564471221974657, : y 절편 
    rvalue=0.8822203446134699, : 설명력
    pvalue=2.8476895206683644e-50, : F검정 
    stderr=0.028577934409305443) : 표준오차 
'''

a = model.slope # x 기울기
b = model.intercept # y 절편 

# 회귀방정식 -> y 예측치 
X = 140; Y = 90 # 1개 관측치 

y_pred = (X*a) + b
print(y_pred) # 88.34388625958358

err = Y - y_pred
print('err=', err) # err= 1.6561137404164157

# 전체 관측치 대상 
len(x) # 150
y_pred = (x*a) + b # 예측치 
len(y_pred) # 150

# 관측치 vs 예측치 
print('관측치 평균 : ', y.mean())
print('예측치 평균 : ', y_pred.mean())

print(y[:10])
print(y_pred[:10])


# 2. 회귀모델 시각화 
import matplotlib.pyplot as plt

# 산점도 
plt.plot(score_iq['iq'], score_iq['score'], 'b.')
# 회귀선 
plt.plot(score_iq['iq'], y_pred, 'r.-')
plt.title('line regression') # 제목 
plt.legend(['x y scatter', 'line regression']) # 범례 
plt.show()



# 3. 다중선형회귀분석 : formula 형식 
from statsmodels.formula.api import ols


# 상관계수 행렬 
corr = score_iq.corr()
print(corr['score'])


obj = ols(formula='score ~ iq + academy + tv', data = score_iq)
model = obj.fit()

# 회귀계수값 반환 
print('회귀 계수값\n%s'%(model.params))

# model의 적합치 
print('model 적합치 :', model.fittedvalues)

# 회귀분석 결과 제공  
print(model.summary()) 




