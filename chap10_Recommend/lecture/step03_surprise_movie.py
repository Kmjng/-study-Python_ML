# -*- coding: utf-8 -*-
"""
- 특이값 분해(SVD) 알고리즘 이용 추천 시스템

<준비물>
 scikit-surprise 패키지 설치 
> conda install -c conda-forge scikit-surprise
"""

import pandas as pd # csv file 
from surprise import SVD # SVD model 
from surprise import Reader, Dataset # SVD dataset 


# 1. dataset loading 
ratings = pd.read_csv('C:/ITWILL/4_Python_ML/data/movie_rating.csv')
print(ratings) #  평가자[critic]   영화[title]  평점[rating]


# 2. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
#  (피벗테이블은 확인용)
print('movie_ratings')
movie_ratings = pd.pivot_table(ratings,
               index = 'title',
               columns = 'critic',
               values = 'rating').reset_index()


# 3. SVD dataset 
reader = Reader(rating_scale=(1, 5)) # 평점 척도 범위 1~5
data = Dataset.load_from_df(ratings, reader) #(원자료, Reader객체)


# 4. train/test set 생성 
# 방법 1
trainset = data.build_full_trainset() # 훈련셋 ★
testset = trainset.build_testset() # 평가셋  ★


# 5. SVD model 생성 
model = SVD(random_state=123).fit(trainset) # seed값 적용 # 훈련셋으로 학습
dir(model)
'''
predict() : 대상자 기준 예측 
test() : 전체 평가셋 예측 
'''

# 6. 전체 평가셋 예측 
all_pred = model.test(testset) # 평가셋으로 예측하기 
print(all_pred)
'''
(uid = '사용자', iid = '아이템', r_ui = 실제평점, est=예측평점)

[Prediction(uid='Jack', iid='Lady', r_ui=3.0, est=3.270719540168945, 
            details={'was_impossible': False}), 
 Prediction(uid='Jack', iid='Snakes', 
            r_ui=4.0, est=3.7291013796093884,
            details={'was_impossible': False}), 
 Prediction(uid='Jack',.....  ]

'''

# 7. Toby 사용자 영화 추천 예측 
user_id  = 'Toby'   # 대상자 
items = ['Just My','Lady','The Night']   
actual_rating = 0 # 실제 등급 0으로 해줌 (기존 실제 등급 NaN)

for item_id in items :
    svd_pred = model.predict(user_id, item_id, actual_rating)
    print(svd_pred)
'''
user: Toby       item: Just My    r_ui = 0.00   est = 2.88   
{'was_impossible': False}
user: Toby       item: Lady       r_ui = 0.00   est = 3.27   
{'was_impossible': False}
user: Toby       item: The Night  r_ui = 0.00   est = 3.30   
{'was_impossible': False}

'''