# -*- coding: utf-8 -*-
"""
step03_DecisionTree_encoding.py

# 의사결정나무는 가중치를 고려하지 않기 때문에 레이블인코딩을 사용한다. 
 Label Encoding 
 - 일반적으로 y변수(대상변수)를 대상으로 인코딩 
 - 트리 계열 모델(의사결정트리, 랜덤포레스트)에서 x변수에 적용
"""

import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 
import matplotlib.pyplot as plt # 중요변수 시각화 

# 1. 화장품 데이터(skin.csv) 가져오기 
df = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\skin.csv")
df.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 제외 
 1   gender       30 non-null     object -> x변수  
 2   age          30 non-null     int64  -> x변수 
 3   job          30 non-null     object -> x변수
 4   marry        30 non-null     object -> x변수
 5   car          30 non-null     object -> x변수
 6   cupon_react  30 non-null     object -> y변수(화장품 구입여부) 
'''
   

# 범주형 변수의 범주(category) 확인 
# 함수 생성
def category_view(df, cols) : 
    for name in cols :
        print('{0} -> {1}'.format(name, df[name].unique())) # df[칼럼명]

category_view(df, df.columns)
'''
cust_no -> [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30]
gender -> ['male' 'female']
age -> [30 20 40]
job -> ['NO' 'YES']
marry -> ['YES' 'NO']
car -> ['NO' 'YES']
cupon_react -> ['NO' 'YES']
'''
# 2. X, y변수 선택 
X = df.drop(['cust_no', 'cupon_react'], axis = 1) # X변수 선택 
y = df['cupon_react'] # y변수 선택 


# 3. data 인코딩 : 문자형 -> 숫자형 

# X변수 인코딩 
# str(obj..) 형태에서는 모델을 학습시킬 수 없다. (에러)
X['gender'] = LabelEncoder().fit_transform(X['gender'])
X['job'] = LabelEncoder().fit_transform(X['job'])
X['marry'] = LabelEncoder().fit_transform(X['marry'])
X['car'] = LabelEncoder().fit_transform(X['car'])

# y변수 인코딩
y = LabelEncoder().fit_transform(y)  

                                                                                     
# 4.훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 123, test_size =0.25)


# 5. model 생성 : DecisionTree 분류기 
model = DecisionTreeClassifier().fit(X_train, y_train) # ValueError
# 인코딩 생략 시 오류 발생(ValueError)


# 6. 중요 변수 
print("중요도 : \n{}".format(model.feature_importances_))
'''
중요도 : 
[0.03459119 0.32806604 0.34591195 0.2132015  0.07822932]
'''
x_size = 5 # x변수 개수
x_names = list(X.columns) # x변수명 추출 
x_names # ['gender', 'age', 'job', 'marry', 'car']
# 중요변수 시각화 : 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) # y = y축 데이터, width = x축 데이터
plt.yticks(range(x_size), x_names) # y축 눈금 : x변수명 적용  
plt.xlabel('feature_importances')
plt.show()



# 7. 모델 평가  
y_pred= model.predict(X_test) # 예측치

# 분류정확도 
accuracy = accuracy_score( y_test, y_pred)
print( accuracy) # 0.875 

# 혼동행렬
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    
'''
[[4 0]
 [1 3]]
'''
# 정밀도 , 재현율, f1 score 확인 ★★★★
report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

(클래스0)  0       0.80      1.00      0.89         4 (빈도수)
(클래스1)  1       1.00      0.75      0.86         4

    accuracy                           0.88         8
   macro avg       0.90      0.88      0.87         8
weighted avg       0.90      0.88      0.87         8


macro avg = 각 평가에 대한 산술평균
=> 0.9 = (0.80 + 1.00)/2
weighted avg = support 를 가중치로 한 가중평균 
=> ((0.8 * 4) + (1.00 *4))/8
'''
