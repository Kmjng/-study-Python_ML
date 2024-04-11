# -*- coding: utf-8 -*-
"""
 연속형 변수 시각화 : 산점도, 히스토그램, box-plot  
 - 연속형 변수 : 셀수 없는 숫자형 변수
   예) 급여, 나이, 몸무게 등   
"""

import random # 난수 생성 
import statistics as st # 수학/통계 함수 
import matplotlib.pyplot as plt # data 시각화 

# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 그래프 자료 생성 
data1 = range(-3, 7) # -3 ~ 6
data2 = [random.random() for i in range(10)] # 0~1사이 난수 실수 
data2

# 1. 산점도 그래프 : 2개 변수 이용, 단일 색상  
plt.scatter(x=data1, y=data2, c='r', marker='o')
plt.title('scatter plot')
plt.show()


# 군집별 산점도 : 군집별 색상 적용 
# cdata에 4가지 색상 랜덤 적용 ★★
cdata = [random.randint(a=1, b=4) for i in range(10)]  # 난수 정수(1~4) 
cdata # [4, 1, 3, 4, 4, 2, 1, 2, 3, 2]

plt.scatter(x=data1, y=data2, c=cdata, marker='o') # c인자에 색상정보
plt.title('scatter plot')
plt.show()


# 군집별 label 추가 
plt.scatter(x=data1, y=data2, c=cdata, marker='o') # 산점도 

for idx, val in enumerate(cdata) : # 색인, 내용 
        # cdata 리스트에 대한 색인과 내용
    plt.annotate(text=val, xy=(data1[idx], data2[idx]))
plt.title('scatter plot') # 제목 
plt.show()



# 2. 히스토그램 그래프 : 1개 변수, 대칭성 확인     
data3 = [random.gauss(mu=0, sigma=1) for i in range(1000)] 
print(data3) # 표준정규분포(-3 ~ +3) 

# 난수 통계
min(data3) # -3.2816997272974784
max(data3) # 3.255899905138625

# 평균과 표준편차 
st.mean(data3) # 0.015763323542712086
st.stdev(data3) # 0.9974539216362369


# 정규분포 시각화 
'''
히스토그램 : x축(계급), y축(빈도수)
'''
help(plt.hist)
plt.hist(data3, label='hist1') # 기본형(계급=10),histtype='bar'  
plt.hist(data3, bins=20, histtype='stepfilled', label='hist2') # 계급, 계단형 적용  
plt.legend(loc = 'best') # 범례
plt.show()
'''
loc 속성 (위치 속성)
- best 
- lower left/right
- upper left/right
- center 
'''


# 3. 박스 플롯(box plot)  : 기초통계 & 이상치(outlier) 시각화
data4 = [random.randint(a=45, b=85) for i in range(100)]  # 45~85 난수 정수 
data4

plt.boxplot(data4)
plt.show()

# 기초통계 : 최솟값/최댓값, 사분위수(1,2,3)
min(data4) # 45
max(data4) # 85

# 사분위수 : q1, q2, q3
st.quantiles(data4) # [54.0, 65.0, 74.75]

st.median(data4) # 중위수 : 65.0
st.mean(data4) # 평균 : 64.98


# 4. 이상치(outlier) 발견 & 처리 
import pandas as pd 

path = r'C:/ITWILL/4_Python_ML/data'

insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()
'''
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
'''

# 1) subset 만들기 
df = insurance[['age','bmi']]
df.shape # (1338, 2)
df
# 2) 이상치 발견과 처리 
df.describe() # # 요약통계량 

# 3) 이상치 시각화 
plt.boxplot(df) # 1: age, 2: bmi 
plt.show()
# age에 이상치들 발견

# 4) age 이상치 처리 : 100세 이하 -> subset 
new_df = df[df['age'] <= 100]

plt.boxplot(new_df)
plt.show()


# 5) bmi 이상치 처리 : iqr 방식  
# 참고로 절대영점을 갖는 비율척도이기 때문에 0 미만의 수 가질 수 없다. 
new_df['bmi'].describe()

q1 = 26.22
q3 = 34.6875
iqr = q3 - q1

outlier_step = 1.5 * iqr
minval = q1 - outlier_step # 13.5187..
maxval = q3 + outlier_step # 47.38875
maxval
new_df=new_df[(new_df['bmi'] <= maxval) & (new_df['bmi'] >= minval)]
plt.boxplot(new_df)
plt.show()






