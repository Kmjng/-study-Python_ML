'''
 문2) iris dataset을 이용하여 다음과 같은 단계로 XGBoost model을 생성하시오.
'''

import pandas as pd # file read
from xgboost import XGBClassifier # model 생성 
from xgboost import plot_importance # 중요변수 시각화  
import matplotlib.pyplot as plt # 중요변수 시각화 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, classification_report # model 평가 
from sklearn.preprocessing import LabelEncoder # y변수 인코딩 

# 단계1 : data set load 
iris = pd.read_csv("C:/ITWILL/4_Python_ML/data/iris.csv")
iris.info()

# 변수명 추출 
cols=list(iris.columns)
col_x=cols[:4] # x변수명 
col_y=cols[-1] # y변수명 

# 단계2 : 훈련/검정 데이터셋 생성
train_set, test_set = train_test_split(iris, test_size=0.25)
train_set.shape # (112, 5)
test_set.shape # (38, 5)

# y변수 인코딩 
train_y = LabelEncoder().fit_transform(train_set[col_y])

# 단계3 : model 생성 : train data 이용
xgb = XGBClassifier(objective='multi:softprob') # softmax 활성함수 
model = xgb.fit(X=train_set[col_x], y=train_y)


# 단계4 :예측치 생성 : test data 이용  
y_pred = model.predict(test_set[col_x])


# 단계5 : 중요변수 확인 & 시각화  
plot_importance(model)
# 꽃잎길이 > 꽃잎넓이 > 꽃받침길이 > 꽃받침넓이 

# 단계6 : model 평가 : confusion matrix, classification_report
y_true = LabelEncoder().fit_transform(test_set[col_y]) # y변수 인코딩 

print(confusion_matrix(y_true, y_pred))
'''
[[11  0  0]
 [ 0 12  1]
 [ 0  1 13]]
'''

report = classification_report(y_true, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        11
           1       0.92      0.92      0.92        13
           2       0.93      0.93      0.93        14

    accuracy                           0.95        38
   macro avg       0.95      0.95      0.95        38
weighted avg       0.95      0.95      0.95        38
'''

















