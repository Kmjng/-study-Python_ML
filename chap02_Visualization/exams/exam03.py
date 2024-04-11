# -*- coding: utf-8 -*-
'''
문3) seaborn의  titanic 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
  <단계1> 'survived','pclass', 'age','fare' 칼럼으로 서브셋 만들기
  <단계2> 'survived' 칼럼을 집단변수로 하여 'pclass', 'age','fare' 칼럼 간의 산점도행렬 시각화
  <단계3> 산점도행렬의 시각화 결과 해설하기  
'''

import matplotlib.pyplot as plt
import seaborn as sn


titanic = sn.load_dataset('titanic')
print(titanic.info())


#  <단계1> 'survived','pclass', 'age','fare' 칼럼으로 서브셋 만들기  
titanic_df = None


# <단계2> 'survived' 칼럼을 집단변수로 하여 'pclass', 'age','fare' 칼럼 간의 산점도 행렬 시각화


# <단계3> 산점도 행렬에서 pclass, age, fare와 survived 변수의 관계 해설


