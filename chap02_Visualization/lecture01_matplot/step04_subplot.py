# -*- coding: utf-8 -*-
"""
  서브플롯(subplot) 차트 시각화 
"""

import random # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 


# 1. subplot 생성 
fig = plt.figure(figsize = (10, 5)) # 차트 크기 지정 
x1 = fig.add_subplot(2,2,1) # 2행2열 1번 
x2 = fig.add_subplot(2,2,2) # 2행2열 2번 
x3 = fig.add_subplot(2,2,3) # 2행2열 3번 
x4 = fig.add_subplot(2,2,4) # 2행2열 4번 

# 2.차트 데이터 생성 
data1 = [random.gauss(mu=0, sigma=1) for i in range(100)] # 정규분포 난수 100개  
data2 = [random.randint(1, 100) for i in range(100)] # 1 ~ 100 난수 정수 100개 
cdata = [random.randint(1, 4) for i in range(100)] # 1 ~ 4


# 3. 각 격차에 차트 크리기 
x1.hist(data1) # 히스토그램 
x2.scatter(data1, data2, c=cdata) # 산점도 
x3.plot(data2) # 기본 차트 
x4.plot(data1, data2, 'g--') # 기본 차트 : 선 색과 스타일 

# subplot 수준 제목 적용 
x1.set_title('hist', fontsize=15)
x2.set_title('scatter', fontsize=15)
x3.set_title('default plot', fontsize=15)
x4.set_title('color plot', fontsize=15)

# figure 수준 제목 적용 
fig.suptitle('suptitle title', fontsize=20)
plt.show()





