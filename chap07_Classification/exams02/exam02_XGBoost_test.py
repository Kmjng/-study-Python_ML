'''
 문2) iris dataset을 이용하여 다음과 같은 단계로 XGBoost model을 생성하시오.
'''

import pandas as pd # file read
from xgboost import XGBClassifier # model 생성 
from xgboost import plot_importance # 중요변수 시각화  
import matplotlib.pyplot as plt # 중요변수 시각화 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, classification_report # model 평가 


# 단계1 : data set load 
iris = pd.read_csv("C:/ITWILL/4_Python_ML/data/iris.csv")

# 변수명 추출 
cols=list(iris.columns)
col_x=cols[:4] # x변수명 
col_y=cols[-1] # y변수명 

col_x # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
col_y # 'Species'


# 단계2 : 훈련/검정 데이터셋 생성 (75% 학습)
train_set, test_set = train_test_split(iris, test_size=0.25)
# train_set에 X, y 변수 함께 들어가있음 # test_set도 
# 나중에 분할해주기 
train_set.shape # (112, 5)
X = train_set.loc[:,col_x] #또는 train_set[col_x]
X.shape # (112, 4)
y = train_set.loc[:,col_y]

# y변수 레이블인코딩 
from sklearn.preprocessing import LabelEncoder 

labels = LabelEncoder().fit_transform(y)
labels

# 단계3 : model 생성 : train data 이용
xgb = XGBClassifier(objective='multi:softprob')
# fit(X,labels)

# 단계4 :예측치 생성 : test data 이용  
X_test = test_set.loc[:, col_x]
y_test = test_set.loc[:, col_y] # 정답 
labels_test = LabelEncoder().fit_transform(y_test) 

eval_set = [(X_test, labels_test)] # 평가셋
type(eval_set[0]) #tuple
model = xgb.fit(X, labels, eval_set = eval_set, eval_metric='merror') 
'''
[0]	validation_0-merror:0.10526
[1]	validation_0-merror:0.10526
[2]	validation_0-merror:0.10526
...
[96]	validation_0-merror:0.13158
[97]	validation_0-merror:0.13158
[98]	validation_0-merror:0.13158
[99]	validation_0-merror:0.13158
'''
y_pred = model.predict(X_test) # 예측치 생성 

con_m = confusion_matrix(labels_test, y_pred)
con_m 
'''
[[11,  0,  0],
[ 1, 10,  1],
[ 0,  3, 12]]
'''

# 단계5 : 중요변수 확인 & 시각화  
report = classification_report(labels_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.92      1.00      0.96        11
           1       0.77      0.83      0.80        12
           2       0.92      0.80      0.86        15

    accuracy                           0.87        38
   macro avg       0.87      0.88      0.87        38
weighted avg       0.87      0.87      0.87        38
'''
plot_importance(model) 
plt.show()


# 단계6 : model 평가 : confusion matrix, accuracy, classification_report
