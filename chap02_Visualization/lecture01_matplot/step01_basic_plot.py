# -*- coding: utf-8 -*-
"""
 - 기본 그래프 그리기 
"""

import matplotlib.pyplot as plt # 시각화 
import random # 난수 생성 


# 차트에서 한글과 음수 부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 1. 그래프 자료 생성 
data = range(-3, 7) # (start, stop-1)
print(data) # [-3, ... ,6]
len(data) # 10


# 2. 기본 그래프 
help(plt.plot)
'''
plot(x, y)        # plot x and y using default line style and color
plot(x, y, 'bo')  # plot x and y using blue circle markers
plot(y)           # plot y using x as index array 0..N-1
plot(y, 'r+')     # ditto, but with red plusses
'''

plt.plot(data) # 선색 : 파랑, 스타일 : 실선 
plt.title('선 색 : 파랑, 선 스타일 : 실선 ')
plt.show() # y축=data, x축=index


# 3. 색상 : 빨강, 선스타일(+)
plt.plot(data, 'r+') # y축=data, x축=index
plt.title('선 색 : 빨강, 선 스타일 : +')
plt.show()


# 4. x,y축 선스타일과 색상 & 마커(circle marker)  
data2 = [random.gauss(0, 1) for i in range(10)]  
plt.plot(data, data2, 'ro') # (x=data, y=data2)
plt.show()









