# -*- coding: utf-8 -*-
"""
문3) california 주택가격을 대상으로 다음과 같은 단계별로 선형회귀모델을 작성하시오.
"""


from sklearn.datasets import fetch_california_housing # dataset load
from sklearn.linear_model import LinearRegression  # model
from sklearn.model_selection import train_test_split # dataset split

from sklearn.preprocessing import scale # 표준화(mu=0, st=1)  
import numpy as np # 로그변환 : np.log1p()  

# 캘리포니아 주택 가격 dataset load 
california = fetch_california_housing()
print(california.DESCR)


# 단계1 : 특징변수(8개)와 타켓변수(MEDV) 선택  
X = california.data
y = california.target


# 단계2 : 데이터 스케일링 : X변수(표준화), y변수(로그변화)   
X_new = scale(X=X)
y_new = np.log1p(y)

# 단계3 : 75%(train) vs 25(test) 비율 데이터셋 split : seed값 적용 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 123)

# 단계4 : 회귀모델 생성
model = LinearRegression().fit(X_train, y_train)

# 단계5 : train과 test score 확인  
y_pred = model.predict(X_test)
y_true = y_test

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(train_score, ":", test_score)
# 0.604714940991568 : 0.6093458386889428
