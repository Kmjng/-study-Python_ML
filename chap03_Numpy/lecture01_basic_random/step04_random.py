
# -*- coding: utf-8 -*-
"""
random 모듈
 - 난수 생성 함수 제공 
"""

import numpy as np # 난수 모듈  
import matplotlib.pyplot as plt # 그래프
import pandas as pd # csv file 

# 1. 난수 실수와 정수  

# 1) 난수 실수 
# random.rand(행,열) : [0,1) 
data = np.random.rand(5, 3) # (행, 열)
print(data)

# 차원 모양
print(data.shape) # (5, 3) 

# 난수 통계
print(data.min()) # 0.08132439151228676
print(data.max()) # 0.9462959398267136
print(data.mean()) # 0.63050758946005

# 2) 난수 정수  
data = np.random.randint(165, 175, size=50) 
print(data)

# 차원 모양
print(data.shape) # (50,)

# 난수 통계
print(data.min()) # 165
print(data.max()) # 174
print(data.mean()) # 169.78



# 2. 이항분포 
dir(np.random)
'''
binomial 이항분포 (이산확률분포)
gamma 감마(연속확률분포)
normal 정규분포(연속확률분포)
poisson 포아송분포(연속확률분포)
seed   시드값(난수 fix)
uniform 균등분포(연속확률분포) 
randn (표준정규분포)
'''
np.random.seed(12)
np.random.binomial(n=1, p=0.5, size=10)  # n=1 베르누이 분포
'''
n: 독립시행
p: 성공확률 
size: 반복횟수(표본수)
'''


# 3. 정규분포
height = np.random.normal(173, 5, 2000) 
print(height) # (2000,)

height2 = np.random.normal(173, 5, (500, 4))
print(height2) # (500, 4)


# 난수 통계
print(height.mean()) # 173.64868062947306
print(height2.mean()) # 173.38566887645658

# 정규분포 시각화 
plt.hist(height, bins=100, density=True, histtype='step')
plt.show()


# 4. 표준정규분포 
# 방법 1. randn()사용
standNormal = np.random.randn(500, 3) # 평균0, 표준편차1 생략
standNormal.shape # (500,3)
print(standNormal.mean()) # -0.04444361993656145

# 방법2. normal 함수 이용 
# (평균,표준편차,(행,열))
standNormal2 = np.random.normal(0, 1, (500, 3)) 
print(standNormal2.mean())


# 정규분포 시각화 
plt.hist(standNormal[:,0], 
         bins=100, density=True, histtype='step', label='col1')
plt.hist(standNormal[:,1], 
         bins=100, density=True, histtype='step', label='col2')
plt.hist(standNormal[:,2], 
         bins=100, density=True, histtype='step',label='col3')
plt.legend(loc='best')
plt.show()


# 5. 균등분포 
uniform = np.random.uniform(10, 100, 1000)
plt.hist(uniform, bins=15, density=True)
plt.show()



# 6. DataFrame sampling

## csv file 가져오기
path = r'C:\ITWILL\4_Python_ML\data'
wdbc = pd.read_csv(path + '/wdbc_data.csv')
print(wdbc.info())


# 1) seed값 적용 
np.random.seed(123)

# 2) pandas sample() 이용  
wdbc_df = wdbc.sample(400)
print(wdbc_df.shape) #  (400, 32)
print(wdbc_df.head())

# 3) training vs test sampling
idx = np.random.choice(a=len(wdbc), size=int(len(wdbc) * 0.7), replace = False)

# training dataset 
train_set = wdbc.iloc[idx]




