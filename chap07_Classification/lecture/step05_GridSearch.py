# -*- coding: utf-8 -*-
"""
step05_SVM_GridSearch.py

 - Grid Search : best parameter 찾기 
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_breast_cancer # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_breast_cancer(return_X_y= True)
X.shape # (569, 30)
y # 0 or 1로 구성

# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 
svc = SVC(C=1.0, kernel='rbf', gamma='scale')

model = svc.fit(X=X_train, y=y_train)


# model 평가 
y_pred = model.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9005847953216374


# 4. 선형 SVM : 선형분류 
obj2 = SVC(C=1.0, kernel='linear')

model2 = obj2.fit(X=X_train, y=y_train)

# model 평가 
y_pred = model2.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc) # accuracy = 0.9707602339181286

'''
커널을 linear로했을 때 정확도가 
rbg보다 7%p 더 높게 나왔다.
'''

###############################
### Grid Search 
###############################

from sklearn.model_selection import GridSearchCV 

# 여러 파라미터를 dict로 넣어서 최적의 매개변수 찾는다.
parmas = {'kernel' : ['rbf', 'linear'],
          'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
          'gamma': ['scale', 'auto']} # dict 정의 

# 5. GridSearch model   
# fit()에 X,y를 넣는다. (분할된 데이터x) ★★★
grid_model = GridSearchCV(model, param_grid=parmas, 
                   scoring='accuracy',cv=5, n_jobs=-1).fit(X, y)
''' 
params_grid = param : 찾을 파라미터 
scoring = 'accuracy' : model 평가 방법 기재 
cv = 5 : 5겹(5등분) 교차검정 (cross-validation)
n_jobs = -1 : cpu 사용 수;
				 -1 : 시스템에서 사용 가능한 모든 코어를 사용
				 ** SVM 모델 학습은 병렬화가 가능한 작업이므로, 
					-1을 설정하여 모든 가능한 코어를 활용하여 학습 속도를 높일 수 있습니다

'''
# 1) Best score (.best_score_)
print('best score =', grid_model.best_score_)
# best score = 0.9631268436578171

# 2) Best parameters(.best_params_)
print('best parameters =', grid_model.best_params_)
'''
best parameters = {'C': 100.0, 'gamma': 'scale', 'kernel': 'linear'}
''' 

# 3) Best Model 만들어보자 
svc = SVC(C=100, kernel='linear', gamma = 'scale')

best_model = svc.fit(X=X_train, y = y_train)
y_pred = best_model.predict(X= X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9766081871345029
