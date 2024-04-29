'''
문5) 연령별, 상품별 구매수량의 평균에 차이가 있는지를 검정하시오.(이원분산분석)

   귀무가설 : 모든 집단은 평균에 차이가 없다.
   대립가설 : 적어도 한 집단에는 평균에 차이가 있다.
'''

import pandas as pd
from statsmodels.formula.api import ols # 이원분산분석모델 생성  
import statsmodels.api as sm # 이원분산분석(Two_way_ANOVA)

# 연령별, 상품별 구매수량 데이터셋  
data = pd.DataFrame({
    '연령': ['20대', '20대', '30대', '30대', '40대', '40대'],
    '상품': ['A', 'B', 'A', 'B', 'A', 'B'],
    '구매수량': [10, 9, 14, 13, 8, 9]
})

data 
'''
   연령 상품  구매수량
0  20대  A    10
1  20대  B     9
2  30대  A    14
3  30대  B    13
4  40대  A     8
5  40대  B     9
'''

# 단계1 : 그룹별 통계(연령별 구매수량 평균)  


# 단계2 : 그룹별 통계(상품 구매수량 평균)  


# 단계3 : 이원분산분석모델 생성
model = None


# 단계4 : 이원분산분석(Two_way_ANOVA)
anova_table = None
print(anova_table)


# 단계5 : 이원분산분석 결과 해설 
'''
[해설] 
'''




