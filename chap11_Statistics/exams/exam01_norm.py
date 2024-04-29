# -*- coding: utf-8 -*-
"""
문1) 스포츠 음료를 만드는 회사가 생산하는 주스는 평균 350ml, 표준편차가 3.2ml인 정규분포를
     따른다고 한다. 품질 검사에서 용량이 340ml 이하 일때 불량이라고 한다. 
     전체 제품 중에서 몇 퍼센트가 불량일까?
"""

# 1. 변수 설정 
x = 340
mu = 350
sigma = 3.2


# 2. 표준화 
z = (x-mu)/sigma 
z # -3.125
#z= abs(z) # 3.125
# p(0<z<3.125)

from scipy import stats # 확률분포 + 검정
import numpy as np 
import matplotlib.pyplot as plt # 확률분포의 시각화 

# 3. z값에 대한 확률 : z분포표 이용 
# norm.cdf ; Cumulative Distribution Function, CDF
p = stats.norm.cdf(z) 
p # 0.0008890252991084321


# 4. 용량이 340ml 이하 일때 불량
percentage = p*100 

print('340ml 이하(불량)일 확률 =', percentage) 
# 0.08890252991083925


