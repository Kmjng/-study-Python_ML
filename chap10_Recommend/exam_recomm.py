# -*- coding: utf-8 -*-
"""
문) food 데이터셋을 대상으로 작성된 피벗테이블(pivot table)을 보고 
'g' 사용자가 아직 섭취하지 않은 음식을 대상으로 추천하는 모델을 생성하고, 
추천 결과를 확인하시오. 
"""

import pandas as pd
from surprise import SVD # SVD model 생성 
from surprise import Reader, Dataset # SVD data set 생성 


# 1. 데이터 가져오기 
food = pd.read_csv('C:/ITWILL/4_Python_ML/data/food.csv')
print(food.info()) #    uid(user)  menu(item) count

# 원자료 확인 
food
'''
    uid   menu  count
0   'b'     우유      2
1   'c'    감자       4
2   'd'  달걀후라이      2
3   'e'     식빵      4
4   'a'     우유      3
..  ...    ...    ...
56  'g'    감자       1
57  'f'  달걀후라이      3
58  'c'  달걀후라이      4
59  'f'     우유      1
60  'g'  달걀후라이      3

[61 rows x 3 columns]
'''

# 2. 피벗테이블 작성  (피벗테이블은 확인용)
ptable = pd.pivot_table(food, 
                        values='count',
                        index='uid',
                        columns='menu', 
                        aggfunc= 'mean')
ptable
'''
menu       감자      달걀후라이        식빵   우유   치킨
uid                                         
'a'   3.000000  3.000000       NaN  3.0  NaN
'b'   1.714286  1.666667       NaN  2.0  NaN
'c'   4.250000  4.500000  5.000000  NaN  NaN
'd'   2.000000  2.000000  2.000000  NaN  2.0
'e'        NaN       NaN  4.166667  5.0  4.0
'f'   2.000000  3.000000  5.000000  1.0  NaN
'g'   1.000000  3.000000       NaN  NaN  NaN  => 관심 대상 

추천할 아이템 : 대상 'menu' => '식빵', '우유', '치킨' 
'''
ptable.info()
ptable['감자']  # 왜안됨 ??...


# 3. rating 데이터셋 생성    
rating = food # 원자료 

reader = Reader(rating_scale = (1,5)) # SVD 데이터셋 만들기 
dataset = Dataset.load_from_df(food, reader)


# 4. train/test set 생성 
# SVD데이터셋.build_full_trainset() 
#  훈련셋객체.build_testset()
trainset = dataset.build_full_trainset()
testset = trainset.build_testset() # 주의 ★★★

# 5. model 생성 : train set 이용 
model = SVD(random_state=123).fit(trainset)
all_pred = model.test(testset) # 평가셋으로 예측하기 
print(all_pred)
'''
[Prediction(uid="'b'", iid='우유', r_ui=2.0, est=2.1807766380435814, 
            details={'was_impossible': False}),
 Prediction(uid="'b'", iid='감자 ', r_ui=1.0, est=1.6680346865440714, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='달걀후라이', r_ui=2.0, est=2.000070278840764, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='감자 ', r_ui=1.0, est=1.6680346865440714, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='달걀후라이', r_ui=2.0, est=2.000070278840764, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='감자 ', r_ui=2.0, est=1.6680346865440714, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='감자 ', r_ui=2.0, est=1.6680346865440714, 
            details={'was_impossible': False}), 
 Prediction(uid="'b'", iid='감자 ', r_ui=2.0, est=1.6680346865440714, 
            details={'was_impossible': False}), 
....'''
# 6. 'g' 사용자 대상 음식 추천 예측 

user_id  = 'g'   # 대상자 
items = ['식빵','우유','치킨']   
actual_rating = 0 

for item_id in items :
    svd_pred = model.predict(user_id, item_id, actual_rating)
    print(svd_pred)

'''
user: g     item: 식빵    r_ui = 0.00   est = 3.35   {'was_impossible': False}
user: g     item: 우유    r_ui = 0.00   est = 3.02   {'was_impossible': False}
user: g     item: 치킨    r_ui = 0.00   est = 3.20   {'was_impossible': False}
'''


