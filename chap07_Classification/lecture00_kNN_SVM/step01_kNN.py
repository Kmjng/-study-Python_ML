# -*- coding: utf-8 -*-
"""
step01_kNN.py

 - 알려진 범주로 알려지지 않은 범주 분류 
 - 유클리드 거리계신식 이용 
"""

from sklearn.neighbors import KNeighborsClassifier # class


# 1.dataset 생성 : ppt.7 참고 
# [단맛, 아삭거림]
grape = [8, 5]   # 포도 - 과일(0)
fish = [2, 3]    # 생성 - 단백질(1)
carrot = [7, 10] # 당근 - 채소(2)
orange = [7, 3]  # 오랜지 - 과일(0)
celery = [3, 8]  # 셀러리 - 채소(2)
cheese = [1, 1]  # 치즈 - 단백질(1)

# x변수 : 알려진 그룹  
know = [grape,fish,carrot,orange,celery,cheese]  # 중첩 list

# y변수 : 알려진 그룹의 클래스 # 지도학습 !!
y_class = [0, 1, 2, 0, 2, 1] 

# 알려진 그룹의 클래스 이름(class name) 
class_label = ['과일', '단백질', '채소'] 
 

# 2. 분류기 
knn = KNeighborsClassifier(n_neighbors = 3) # k=3 
model = knn.fit(X = know, y = y_class) 
know # [[8, 5], [2, 3], [7, 10], [7, 3], [3, 8], [1, 1]]

# 3. 분류기 평가 
x1 = 4 # 단맛(1~10) : 4 -> 8 -> 2
X2 = 8 # 아삭거림(1~10) : 8 -> 2 -> 3

# 분류대상 
unKnow = [[x1, X2]]  # 중첩 list : [[4, 8]] [4,8]이 [2]에 매치됨 # '채소'
unKnown = [[[4,8]],[[8,2]],[[2,3]]]
# class 예측 
y_pred = model.predict(X = unKnow)
print(y_pred) # [2] -> [0] -> [1]

print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 채소
print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 과일
print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 단백질

for i in unKnown:
    y_pred = model.predict(X=i)
    print('분류결과 : ',class_label[y_pred[0]]) # 분류결과 : 채소/과일/단백질



# know(6) vs unKnow(1) 

import numpy as np 

# 1. 다차원 배열 변환 
know_arr = np.array(know) # 중첩리스트 
# [[8, 5], [2, 3], [7, 10], [7, 3], [3, 8], [1, 1]]
unKnow_arr = np.array(unKnow) # 중첩리스트
# [[x1, X2]] #[[4,8]]

know_arr.shape # (6, 2)
unKnow_arr.shape # (1, 2)

# 유클리드 거리계산식 : 차(-) -> 제곱 -> 합 -> 제곱근 
diff = know_arr - unKnow_arr # 1) 차(-)
diff

diff_square = np.square(diff) # 2) 제곱
diff_square

diff_square_sum = diff_square.sum(axis = 1) # 3) 열끼리 합
diff_square_sum # [25, 29, 13, 34,  1, 58]

distance = np.sqrt(diff_square_sum) # 4) 제곱근 

print(distance) # k=3
# [5.  5.38516481 3.60555128 5.83095189 1. 7.61577311]
# 새로운 데이터 unKnow 와 기존 6개 데이터 포인트 간의 최근접 거리 

# 오름차순 색인정렬 : k=3
# 인덱스 반환 함수(배열) argsort()  
idx = distance.argsort()[:3] # [4, 2, 0, 1, 3, 5] 
# 여기서 4는 distance 1의 인덱스!!
idx # [4, 2, 0]

y_class = [0, 1, 2, 0, 2, 1] # (위에서 선언됨)

for i in idx :
    #print(y_class[i])
    print(class_label[y_class[i]])
'''
채소  # y_class[4]가 채소였음
채소  # y_class[2]도 채소였음
과일  # y_class[0]은 과일이었음

토마토는 채소, 과일에 가깝다 ~
'''
