# -*- coding: utf-8 -*-
"""
statistics 모듈의 주요 함수 
  - 기술통계 : 대푯값, 산포도, 왜도/첨도 등   
"""

import statistics as st # 기술통계 
import pandas as pd # csv file 


# 기술통계 
path = r'C:\ITWILL\4_Python_ML\data'
dataset = pd.read_csv(path + '/descriptive.csv')
print(dataset.info())

x = dataset['cost'] # 구매비용 선택 

# 1. 대푯값
print('평균 =', st.mean(x)) 
print('중위수=', st.median(x)) 
print('낮은 중위수 = ', st.median_low(x)) # 중위수 이하의 중위수
print('높은 중위수 = ', st.median_high(x))# 중위수 이상의 중위수
print('최빈수 =',  st.mode(x)) 


# 2. 산포도   
var = st.variance(x)
print('표본의 분산 = ', var) 
print('모집단의 분산 =', st.pvariance(x)) 

std = st.stdev(x)
print('표본의 표준편차 =', std) 
print('모집단의 표준편차 =', st.pstdev(x))

# 사분위수 
print('사분위수 :', st.quantiles(x)) 


(160-162)/5 # -0.4(0.1554)

0.5 - 0.1554 # 0.3446


import scipy.stats as sts

# 3. 왜도/첨도 

# 1) 왜도 
sts.skew(x) # -0.1531779106237012 -> 오른쪽으로 기울어짐 
'''
왜도 = 0 : 좌우대칭 
왜도 > 0 : 왼쪽 치우침
왜도 < 0 :  오른쪽 치우침
'''

# 첨도 
sts.kurtosis(x) # -0.1830774864331568 = fisher 기준 
sts.kurtosis(x, fisher=False) # 2.816922513566843 = Pearson 기준 
'''
첨도 0 : 정규분포 첨도 
첨도 > 0 : 뾰족함
첨도 < 0 : 완만함  

피어슨 첨도공식 = 피셔 첨도공식 +3 
정규분포 첨도가 
피어슨 기준 kurtosis = 3
피셔 기준 kurtosis = 0 
'''

