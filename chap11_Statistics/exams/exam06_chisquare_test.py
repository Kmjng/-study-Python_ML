# -*- coding: utf-8 -*-
"""
문6) titanic 데이터셋을 이용하여 다음과 같이 카이제곱검정을 수행하시오.
   <단계1> 생존여부(survived), 사회적지위(pclass) 변수를 이용하여 교차분할표 작성 
   <단계2> 카이제곱 검정통계량, 유의확률, 자유도, 기대값 출력      
   <단계3> 가설검정 결과 해설  
   
   귀무가설 : 생존여부(survived)와 사회적지위(pclass)는 관련성이 없다. 
"""

import seaborn as sn
import pandas as pd
from scipy import stats # 확률분포 검정 

# titanic dataset load 
titanic = sn.load_dataset('titanic')
print(titanic.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
...
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
None
'''
# <단계1> 교차분할표 
pclass = titanic.pclass
survived = titanic.survived 
tab = pd.crosstab(index = pclass, columns = survived,
                  margins =True) # margin =True : '총(All)' 추가
tab

'''
survived    0    1  All
pclass                 
1          80  136  216
2          97   87  184
3         372  119  491
All       549  342  891
'''
# <단계2> 카이제곱 검정통계량, 유의확률, 자유도, 기대값  
chi2, pvalue, df, evalue = stats.chi2_contingency(observed= tab)
print('카이제곱검정통계량:%.2f\npvalue: %f\n자유도:%d'%(chi2, pvalue, df))
'''
카이제곱검정통계량:102.89
pvalue: 0.000000
자유도:6
'''
# 기댓값 
print(evalue)
'''
[[133.09090909  82.90909091 216.        ]
 [113.37373737  70.62626263 184.        ]
 [302.53535354 188.46464646 491.        ]
 [549.         342.         891.        ]]
'''
# <단계3> 가설검정 해설 


