'''
문3) 다음과 같은 wine 자료를 이용하여 red 와인과 white 와인의 
alcohol에 대해서 
     유의수준 5%에서 독립표분 t검정을 수행하시오.
'''

import pandas as pd
from scipy import stats


path = r'C:\ITWILL\4_Python_ML\data'
wine = pd.read_csv(path + '/winequality-both.csv')
print(wine.info())
'''
0   type                  6497 non-null   object   : 와인 유형 
1   fixed acidity         6497 non-null   float64
2   volatile acidity      6497 non-null   float64
3   citric acid           6497 non-null   float64
4   residual sugar        6497 non-null   float64
5   chlorides             6497 non-null   float64
6   free sulfur dioxide   6497 non-null   float64
7   total sulfur dioxide  6497 non-null   float64
8   density               6497 non-null   float64
9   pH                    6497 non-null   float64
10  sulphates             6497 non-null   float64
11  alcohol               6497 non-null   float64  : 알콜량
12  quality               6497 non-null   int64    
'''


# 1. red 와인과 white 와인의 alcohol에 대한 subset 만들기  
red_alcohol = wine.loc[wine.type=='red', 'alcohol']
white_alcohol = wine.loc[wine.type=='white', 'alcohol']
red_alcohol

# 2. 기술통계 : 각 집단별 평균
red_alcohol.mean() # 10.422983114446529
white_alcohol.mean() # 10.51426704777011


# 3. 독립표본 t검정 : 두 집단 평균 검정
two_sample = stats.ttest_ind(red_alcohol, white_alcohol)

print('t검정 통계량 = %.3f, pvalue = %.5f'%(two_sample))

# 4. 검정결과 해설 
'''
t검정 통계량 = -2.659, pvalue = 0.00787

>> 유의수준 0.05에 의해 두 집단 평균 차이가 있다. 
'''
