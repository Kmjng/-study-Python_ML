# -*- coding: utf-8 -*-
"""
- 이산형 변수 시각화 : 막대 그래프, 원 그래프 
- 이산형 변수 : 셀수 있는 숫자형 변수 
   예) 가족수, 자녀수, 자동차 대수 등 
"""

import matplotlib.pyplot as plt  

# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 그래프 자료 생성 
data = [127, 90, 201, 150, 250] # 국가별 수출현황 
year = [2010, 2011, 2012, 2013, 2014] # 년도 


# 1. 세로막대 그래프 
plt.bar(x = year, height=data) # 기본색상  
plt.title('국가별 수출현황')
plt.xlabel('년도별')
plt.ylabel('수출현황(단위 : 달러)')
plt.show()


# 2. 가로막대 그래프
plt.barh(y= year, width = data, color='blue') # 색상적용  
plt.title('국가별 수출현황')
plt.xlabel('수출현황(단위 : 달러)')
plt.ylabel('년도별')
plt.show()


# 3. 누적형 세로막대 그래프
cate = ['A', 'B', 'C', 'D'] # 집단 
val1 = [15, 8, 12, 10]  # 첫 번째 데이터셋
val2 = [5, 12, 8, 15]  # 두 번째 데이터셋

plt.bar(cate, val1, label='데이터셋1', alpha=0.5) # 투명도  
plt.bar(cate, val2, bottom=val1, label='데이터셋2', alpha=1)
# bottom : val1 위에 val2 올리기  
plt.title('누적형 막대 그래프')
plt.xlabel('카테고리')
plt.ylabel('값')
plt.legend() # 범례 추가
plt.show()



# 4. 원 그래프 : 비율 적용 
labels = ['싱가폴','태국','한국','일본','미국'] 
print(data) # [127, 90, 201, 150, 250]

plt.pie(x = data, labels = labels) # 100% 
plt.show()


# 비율 계산 
tot = sum(data)
rate = [(d / tot)*100  for d in data ] # 백분율   

# 새로운 lable 
new_lables = [] 

for i in range(len(labels)) :
    new_lables.append(labels[i] + '\n' + str(rate[i]) + '%')
    
new_lables
plt.pie(x = data, labels = new_lables) 
plt.show()
















