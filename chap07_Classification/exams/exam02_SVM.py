# -*- coding: utf-8 -*-
"""
문2) 다음 조건에 맞게 비선형 SVM 모델과 선형 SVM 모델을 생성하시오. 
  <조건1> 비선형 SVM 모델과 선형 SVM 모델 생성
  <조건2> GridSearch model을 이용하여 best score와 best parameters 구하기  
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_iris(return_X_y= True)
X.shape # (569, 30)

# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 : kernel='rbf'
rbf_model = SVC(kernel= 'rbf').fit(X=X_train, y=y_train)
y_pred= rbf_model.predict(X=X_test)
y_true = y_test
acc = accuracy_score(y_true, y_pred)
acc # 0.9111,..

# 4. 선형 SVM 모델 : kernel='linear' 
linear_model = SVC(kernel= 'linear').fit(X=X_train, y=y_train)
y_pred = linear_model.predict(X_test)
y_true = y_test 
acc = accuracy_score(y_true, y_pred)
acc # 0.9555,...

# rbf모델이 분류정확도가 더 낮으니 교차검정을 수행해본다. 
# 5. Grid Search : 선형과 비선형 SVM 모델 중 분류정확도가 낮은 model 대상으로 5겹 교차검정 수행 
from sklearn.model_selection import GridSearchCV
params = {'kernel' : ['rbf', 'linear'],
          'C' : [0.01, 0.1, 1.0, 10, 100],
          'gamma': ['scale', 'auto']} # dict 정의 
grid_model = GridSearchCV(rbf_model, param_grid = params, 
                          scoring='accuracy',cv=5, n_jobs=-1).fit(X,y)
b_score = grid_model.best_score_
b_score # 0.9800..
b_params = grid_model.best_params_
b_params
# {'C': 1.0, 'gamma': 'scale', 'kernel': 'linear'}


