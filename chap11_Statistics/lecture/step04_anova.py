"""
분산분석(ANOVA)
 - 세  집단 이상의 평균차이 검정 
"""

import numpy as np
import pandas as pd

from scipy import stats # 일원분산분석(One-way_ANOVA) 
from statsmodels.formula.api import ols # 이원분산분석모델 생성  
import statsmodels.api as sm # 이원분산분석(Two_way_ANOVA) 


### 1. 일원분산분석(One-way_ANOVA) : y(수치형) ~ x(명목형) 
'''
가정 : 세 개의 그룹 (group1, group2, group3)을 생성하고, 이들 간의 평균 차이를 비교한다. 
기본가설 : 세 그룹의 평균은 차이가 없다.
'''

# 세 그룹의 만족도 데이터셋 
group1 = np.array([4, 6, 8, 10, 8])
group2 = np.array([2, 5, 7, 9, 4])
group3 = np.array([1, 3, 5, 2, 4])

 
# 귀무가설(H0) : 세 그룹의 만족도의 평균은 차이가 없다.
# 대립가설(H1) : 적어도 한 집단 이상에서 만족도의 평균에 차이가 있다. 

# 일원분산분석
# stats.f_oneway(group1, group2, group3) 
f_statistic, p_value = stats.f_oneway(group1, group2, group3) 

# 결과 출력
print("F-statistic:", f_statistic)
# F-statistic: 4.4399999999999995
print("p-value:", p_value)
# p-value: 0.03603333923006122
# 대립가설 채택 

# 기술통계 : 사후검정 
group1.mean()  # 7.2
group2.mean()  # 5.4
group3.mean()  # 3.0

'''
사후검정(post hoc test)
통계학에서 실험 또는 연구 결과를 분석할 때 사용되는 기술
실험에서 세 개 이상의 그룹 간에 통계적으로 유의미한 차이가 있는지 확인

Tukey's HSD(Honestly Significant Difference), 
Bonferroni correction, 
Scheffé test 
'''



### 2. 이원분산분석(Two_way_ANOVA) : y ~ x1 + x2
# 귀무가설(H0): 두 개 이상의 집단 간의 평균이 서로 동일하다.
'''
가정 : 약품종류 : A, B, C의 그룹에서 1시간, 2시간 간격으로 측정한 결과에 대한  
      반응 시간에 차이를 검정한다.
독립변수1: 약품종류(A, B, C)
독립변수2: 측정시간(1시간, 2시간)
종속변수: 반응시간
'''

# 귀무가설(H0) : 약품종류와 측정시간은 반응시간에 대해서 집단간 평균의 차이는 없다.  

data = pd.DataFrame({
    'type': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], # 약품종류
    'time': [1, 2, 1, 2, 1, 2, 1, 2, 2],  # 측정시간 
    'retime': [10, 15, 9, 8, 14, 11, 11, 16, 20] # 반응시간 
})



# 이원분산분석 모델 생성 
model = ols('retime ~ type + time', data=data).fit()

# 이원분산분석 
anova_table = sm.stats.anova_lm(model)

# 결과 출력
print(anova_table)
'''
           df     sum_sq    mean_sq         F    PR(>F)
type      2.0  40.666667  20.333333  1.561433  0.297270  
time      1.0  14.222222  14.222222  1.092150  0.343864
Residual  5.0  65.111111  13.022222       NaN       NaN

F-value = WV/BV  (=MS1/MS2) (=MSR/MSE)
PR(>F) 가 F통계량에 대한 pvalue 임 

해설: 측정시간(time)은 반응시간에 대해 집단 간 평균차이가 없다. 
약품종류(type)은 반응시간에 대해 집단간 평균차이가 없다.
'''

