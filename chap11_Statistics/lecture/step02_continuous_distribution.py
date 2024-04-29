# -*- coding: utf-8 -*-
"""
step02_continuous_distribution.py

확률분포와 검정(test)

1. 연속확률분포와 정규성 검정 
  - 연속확률분포 : 정규분포, 균등분포, 카이제곱, T/Z/F 분포 등 
  - 정규분포의 검정   
2. 이산확률분포와 이항 검정 
  - 이산확률분포 : 베르누이분포, 이항분포, 포아송분포 등  
  - 이항분포의 검정 
"""

from scipy import stats # 확률분포 + 검정
import numpy as np 
import matplotlib.pyplot as plt # 확률분포의 시각화 


################################
### 정규분포와 검정(정규성 검정)
################################


# 단계1. 모집단에서 표본추출  

# 평균 및 표준 편차 설정
mu = 0
sigma = 1

# 표준정규분포에서 표본추출 : 확률변수 X 
X = stats.norm.rvs(loc=mu, scale=sigma, size=1000)  
'''
rvs(random variable sampling) : N개 표본추출 

loc : 모평균 
scale : 모표준편차 
size : 표본 크기 

'''
X.shape #(1000,)


# 단계2. 확률밀도함수(pdf)    

# '밀도곡선'을 위한 벡터 자료   
# x축 데이터 ; np.linspace() 
line = np.linspace(min(X), max(X), 100) # 밀도곡선 
line.shape # (100,) # min(X)~max(X) 를 100개의 구간으로 나눔 

# 히스토그램 : 단위(밀도)
# pdf ; 확률밀도함수 
plt.hist(X, bins='auto', density=True)  
#plt.plot(x축, y축)
plt.plot(line, stats.norm.pdf(line, mu, sigma), color='red') 
plt.show()


# 단계3. 정규성 검정 
# stats.shapiro(X) ; X는 위에서 추출한 확률분포
''' 
 귀무가설(H0) : 정규분포와 차이가 없다. (검정 목적과 반대되는 주장으로)
 대립가설(H1) : 정규분포와 차이가 있다.
'''
print(stats.shapiro(X))

statistic, pvalue = stats.shapiro(X)
print('검정통계량 = ', statistic)
'''
검정통계량 =  0.9984564185142517 (0에 가까울수록 정규분포)
        정규분포 기댓값(0)과 얼마나 떨어져 있는지 ★★★
p-value =  0.5265042781829834   (확률값)
'''

alpha = 0.05 # 유의수준(알파) ; 연구자가 설정 
'''
pvalue > alpha : 가설 채택 
pvalue < alpha : 가설 기각 
'''
if pvalue > alpha : 
    print('정규분포와 차이 없다. 귀무가설채택')
else :
    print("정규분포와 다르다. 대립가설채택")




#######################################
## 정규분포와 표준정규분포
#######################################

# 단계1. 모집단에서 정규분포 추출   
'''
성인여성(19~24세)의 키는 평균이 162cm, 표준편차가 5cm인 정규분포
'''
np.random.seed(45)


# 모평균과 모표준편차 설정
mu = 162  # 평균키 
sigma = 5 # 표준편차 


# 정규분포에서 표본추출 : 확률변수 X 
X = stats.norm.rvs(loc=mu, scale=sigma, size=1000)  


# 단계2. 확률밀도함수(pdf)    

# 밀도곡선을 위한 벡터 자료   
line = np.linspace(min(X), max(X), 100)

# 히스토그램 : 단위(밀도)
plt.hist(X, bins='auto', density=True)  
plt.plot(line, stats.norm.pdf(line, mu, sigma), color='red') 
plt.show()


# 단계3. 정규성 검정 
statistic, pvalue = stats.shapiro(X)
print('검정통계량 = ', statistic) 
print('p-value = ', pvalue) 
'''
검정통계량 =  0.9991166591644287
p-value =  0.9262917637825012
'''

# 단계4. 표준정규분포  
sten_norm = (X - mu) / sigma 
type(sten_norm) # numpy
line = np.linspace(min(sten_norm), max(sten_norm), 100)
plt.hist(sten_norm, bins='auto', density= True)
plt.plot(line, stats.norm.pdf(line, 0, 1), color = 'red')
plt.show()

'''
가정 : ppt.31
   성인여성(19~24세)의 키는 평균이 162cm, 표준편차가 5cm인 정규분포를 
   따른다고 한다. 성인 여자의 키가 160cm 이하인 비율은 얼마일까?
'''

# 변수 설정 
x = 160
mu = 162
sigma = 5

# 표준화 
z = (x - mu) / sigma # -0.4
# z값에 대한 확률 : z분포표 통해 해당 확률 
p = 0.1554 # p(0 < z < 0.4) = 0.1554

# 160cm 이하 비율 : 좌우 대칭분포=0.5+0.5=1
result = 0.5 - p # 0.3446
print('160cm 이하인 비율 =', result * 100) # 160cm 이하인 비율 = 34.46



