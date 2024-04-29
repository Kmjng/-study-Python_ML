'''
문4) 10명을 대상으로 광고노출전태도와 광고노출후태도를 점수화한 결과이다.  
    광고에 효과가 있는지를 유의수준 5%에서 대응표본 t검정을 수행하시오. 

 H0 : µd = 0(광고에 효과가 없다.) 
 H1 : µd < 0(광고노출후태도가 증가한다.) 
 
 µd = 광고노출전태도 - 광고노출후태도
'''

import numpy as np 
from scipy import stats
 
pre_exposure = np.array([50,25,30,50,60,80,45,30,65,70]) # 광고 노출전 태도 
post_exposure = np.array([53,27,38,55,61,85,45,31,72,78]) # 광고 노출후 태도 


# 1. µd = 광고노출전태도 - 광고노출후태도


# 2. 대응표본 t검정
paired_sample = None
print('t검정 통계량 = %.5f, pvalue = %.5f'%paired_sample)


# 3. 검정결과 해설 


