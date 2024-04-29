'''  
문7) irsi.csv 데이터셋을 이용하여 단계별로 다중선형회귀모델을 생성하시오.
   단계1 : 칼럼명에 포함된 '.' 을 '_'로 수정   
      iris.columns = iris.columns.str.replace('.', '_')   
   단계2 : model의 formula 구성 
      y변수 : 1번째 칼럼, x변수 : 2 ~ 3번째 칼럼       
   단계3 : 회귀계수 확인    
   단계4 : 회귀모델 결과 확인 및 해석  
'''

import pandas as pd
from statsmodels.formula.api import ols # 다중회귀모델

path = r'c:\itwill\4_python_ml\data'


# dataset 가져오기  
iris = pd.read_csv(path + '/iris.csv')
print(iris.head())

# 단계1. iris 칼럼명 수정 

# 단계2. formula 구성 및 다중회귀모델 생성  

# 단계3. 회귀계수 확인 

# 단계4. 회귀모델 결과 확인 및 해석 : 힌트) :  summary()함수 이용 

