# -*- coding: utf-8 -*-
"""
문6) titanic 데이터셋을 이용하여 다음과 같이 카이제곱검정을 수행하시오.
   <단계1> 생존여부(survived), 사회적지위(pclass) 변수를 이용하여 교차분할표 작성 
   <단계2> 카이제곱 검정통계량, 유의확률, 자유도, 기대값 출력      
   <단계3> 가설검정 결과 해설  
"""

import seaborn as sn
import pandas as pd
from scipy import stats # 확률분포 검정 

# titanic dataset load 
titanic = sn.load_dataset('titanic')
print(titanic.info())

# <단계1> 교차분할표 

# <단계2> 카이제곱 검정통계량, 유의확률, 자유도, 기대값  

# <단계3> 가설검정 해설 


