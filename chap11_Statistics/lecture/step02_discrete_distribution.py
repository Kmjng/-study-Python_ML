# -*- coding: utf-8 -*-
"""
step02_discrete_distribution.py

2. 이산확률분포와 이항 검정 
  - 이산확률분포 : 베르누이분포, 이항분포, 포아송분포 등  
  - 이항분포와 검정 
"""

from scipy import stats # 확률분포 + 검정
import numpy as np # 성공횟수


################################
### 이항분포와 검정
################################
'''
 - 이항분포 : 2가지 범주(성공 or 실패)를 갖는 이산확률분포
 - 베르누이시행 : 이항변수(성공=1 or 실패=0)에서 독립시행 1회 -> 
 - 베르누이분포 : 베르누이시행으로 추출된 확률분포   
 - 이항분포 : 베루누이시행 n번으로 추출된 확률분포
'''
 

# 단계1. 표본 추출(random sampling) 

# 1) 동전 확률실험 : 베르누이분포 모집단에서 표본 추출
sample1 = stats.bernoulli.rvs(p=0.5, size=10) 


# 2) 동전 확률실험 : 이항분포 모집단에서 표본 추출 
sample2 = stats.binom.rvs(n=1, p=0.5, size=10) # 독립시행=1회 
# n =1 ; 위의 베르누이시행과 동일함 
sample2
# [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
sample3 = stats.binom.rvs(n=5, p=0.5, size=10) # 독립시행=5회 
sample3 
# [2, 1, 3, 3, 2, 0, 4, 4, 2, 3]
# 성공두번, 성공한번, 성공세번,...

# [문제] 주사위 확률실험 : 베르누이 독립시행 10회와 성공확률 1/6을 갖는 50개 표본 추출하기  
# 성공 기준 : 주사위 6
sample4 = stats.binom.rvs(n=10, p=1/6, size = 50)
sample4
'''
[1, 0, 2, 1, 1, 1, 2, 2, 2, 2, 3, 1, 3, 2, 4, 2, 3, 1, 0, 3, 2, 0,
       2, 4, 2, 2, 1, 2, 0, 4, 0, 2, 3, 1, 2, 1, 2, 3, 1, 3, 1, 2, 2, 2,
       1, 2, 2, 3, 2, 3] # 각각 10번의 독립시행에서 6이 나온 횟수들 
'''

# 단계2. 이항검정(binom test) : 이항분포에 대한 가설검정 
'''
연구환경 : 게임에 이길 확률(p)이 40%이고, 게임의 시행회수가 50 일 때 95% 신뢰수준에서 검정 

차이가 있다/없다 => 양측검정 ★★★

귀무가설(H0) : 게임에 이길 확률(p)는 40%와 차이가 없다.(p = 40%)
이항검정 => result = stats.binomtest(k=k, n=50, p=0.4, alternative='two-sided') 
대립가설(H1) : 게임에 이길 확률(p)는 40%와 차이가 있다.(p != 40%)
'''

np.random.seed(123) # 동일한 표본 추출  


# 1) 베르누이분포 : B(1, p)에서 표본추출(100개) 
p = 0.4 # 모수(p) : 성공확률 

# 베르누이분포 표본 추출 
binom_sample = stats.binom.rvs(n=1, p=p, size=50)


# 2) 성공횟수 반환 : zero 제외  
# np.count_nonzero() ★★★
print('binom 성공횟수 =', np.count_nonzero(binom_sample)) 
# binom 성공횟수 = 18


# 3) 유의확률 구하기  
k = np.count_nonzero(binom_sample) # 성공(1) 횟수 
k # 18

# 4) 가설검정 
## 이항검정 수행 : 양측검정   
# stats.binomtest() ★★★
result = stats.binomtest(k=k, n=50, p=0.4, alternative='two-sided') 
pvalue = result.pvalue 
pvalue # 0.6654749478818358


# 이항검정 결과     
alpha = 0.05  

if pvalue > alpha :  
    print(f"p-value({pvalue}) : 게임에 이길 성공률 40%와 차이가 없다.")
else:
    print(f"p-value({pvalue}) : 게임에 이길 성공률 40%와 차이가 있다.")



######################### 
# 이항검정 적용 사례   
#########################
'''
연구환경 :  
  남녀 전체 150명의 합격자 중에서 남자 합격자가 62명일 때 99% 신뢰수준에서 
  남여 합격률에 차이가 있다고 할수 있는가?

귀무가설(H0) : 남여 합격률에 차이가 없다.(p = 0.5) ; 남자합격률 p ★★★
대립가설(H0) : 남여 합격률에 차이가 있다.(p != 0.5)
'''


# 이항검정 수행  (양측검정)
k = 62 # 성공횟수 (남자 합격자 수) 
result = stats.binomtest(k=k, n=150, p=0.5, alternative='two-sided')  
pvalue = result.pvalue
pvalue # 0.04086849386649401

print('## 이항 검정 ##')
alpha = 0.05 # 1 -0.025 -0.025 = 1-alpha

if pvalue > 0.05 : # 유의확률 > 유의수준 
    print(f"p-value({pvalue}) >= 0.05 : 남여 합격률에 차이가 없다.")
else:
    print(f"p-value({pvalue}) < 0.05 : 남여 합격률에 차이가 있다.")
    

# 이항검정 수행  (단측검정)
# 귀무가설 : 남자합격자 > 여자합격자 : greater 
# 귀무가설 : 남자합격자 < 여자합격자 : less 
result = stats.binomtest(k=k, n=150, p=0.5, alternative = 'greater')
pvalue = result.pvalue 
pvalue # 0.9864237046311769
alpha= 0.05

if pvalue > 0.05 : 
    print('귀무가설 채택; 남자합격자가 더 많다.')
else: 
    print("귀무가설 기각 ; 남자합격자가 더 많지 않다.")



