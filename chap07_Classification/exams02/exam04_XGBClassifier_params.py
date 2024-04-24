# -*- coding: utf-8 -*-
"""
문4) wine dataset을 이용하여 다음과 같이 다항분류 모델을 생성하시오. 
   <조건1> tree model 200개 학습
   <조건2> tree model 학습과정에서 조기 종료 100회 지정
   <조건3> model의 분류정확도와 리포트 출력   
"""

from xgboost import XGBClassifier # model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine # 다항분류
from sklearn.metrics import classification_report


#################################
## 1. XGBoost Hyper Parameter
#################################

# 1. dataset load
wine = load_wine()
wine.DESCR

# 2. train/test 생성 
X = wine.data 
y = wine.target # 인코딩 되어 있는 듯 

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3)

# 3. model 생성 : 다항분류 

xgb = XGBClassifier(objective='multi:softprob', 
                    n_estimators=200) # softmax 함수 
eval_set = [(X_test, y_test)]
model = xgb.fit(X_train, y_train, eval_metric='merror',
                eval_set = eval_set, 
                verbose=True)
'''
[0]	validation_0-merror:0.01852
[1]	validation_0-merror:0.00000
[2]	validation_0-merror:0.03704
...
[197]	validation_0-merror:0.05556
[198]	validation_0-merror:0.05556
[199]	validation_0-merror:0.05556
'''

# 4. model 학습 조기종료 
# # early stop한 모델 
e_model = xgb.fit(X_train, y_train, eval_metric='merror',
                early_stopping_rounds=100,
                eval_set = eval_set, 
                verbose=True)
'''
[106]	validation_0-merror:0.05556
[107]	validation_0-merror:0.05556
[108]	validation_0-merror:0.05556
[109]	validation_0-merror:0.05556
[110]	validation_0-merror:0.05556
'''

# softmax 확률 확인해보기 
# 각 class 확률합 =1 
y_pred_proba = e_model.predict_proba(X_test) # 확률 
y_pred_proba.shape # (54, 3)
y_pred_proba.sum(axis =1) # 1.0 

'''
array([[0.08400054, 0.832282  , 0.0837174 ], 
       [0.75440496, 0.15381901, 0.09177604],
       [0.15939716, 0.76377624, 0.07682656],
       [0.13017935, 0.74502176, 0.12479887],
       [0.82712203, 0.08657614, 0.08630186],
'''


# 5. model 평가 : classification_report
y_pred1 = model.predict(X_test)
y_pred2 = e_model.predict(X_test) # early stop한 모델 

report1 = classification_report(y_test, y_pred1)
report2 = classification_report(y_test, y_pred2)
print(report1)
'''
              precision    recall  f1-score   support

           0       0.95      1.00      0.97        18
           1       1.00      0.90      0.95        29
           2       0.78      1.00      0.88         7

    accuracy                           0.94        54
   macro avg       0.91      0.97      0.93        54
weighted avg       0.95      0.94      0.95        54
'''
print(report2)
'''
              precision    recall  f1-score   support

           0       0.95      1.00      0.97        18
           1       1.00      0.90      0.95        29
           2       0.78      1.00      0.88         7

    accuracy                           0.94        54
   macro avg       0.91      0.97      0.93        54
weighted avg       0.95      0.94      0.95        54
'''

# 분류정확도 
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2) # early stop한 모델 
print(acc1) # 0.9444444444444444
print(acc2) # 0.9444444444444444
