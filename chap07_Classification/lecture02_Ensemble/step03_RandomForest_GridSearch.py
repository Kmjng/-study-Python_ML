# -*- coding: utf-8 -*-
"""
 1. RandomForest Hyper parameters
 2. GridSearch : best parameters 
"""

from sklearn.ensemble import RandomForestClassifier # model 
from sklearn.datasets import load_digits # dataset 
from sklearn.model_selection import train_test_split # dataset split 

# 1. dataset load
X, y = load_digits(return_X_y=True)


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=15) # 90% 훈련셋 


# 3. 기본 model 생성 
#help(RandomForestClassifier)
model = RandomForestClassifier(random_state=234) # default 적용 

# model 학습 
model.fit(X = X_train, y = y_train) 

# model 평가 
model.score(X = X_test, y = y_test) # 0.96111


'''
주요 hyper parameter(default) 
n_estimators=100 : 결정 트리 개수, 많을 수록 성능이 좋아짐
criterion='gini' : 중요변수 선정기준 : {"gini", "entropy"}
max_depth=None : min_samples_split의 샘플 수 보다 적을 때 까지 tree 깊이 생성
min_samples_split=2 : 내부 node 분할에 사용할 최소 sample 개수
max_features='auto' : 최대 사용할 x변수 개수 : {"auto", "sqrt", "log2"}
 ★★★ => Random Forest 모델에서 각 트리가 분할할 때 고려할 최대 특성의 개수 ★★★
min_samples_leaf=1 : leaf node를 만드는데 필요한 최소한의 sample 개수
n_jobs=None : cpu 사용 개수

********* 주의 ********
의사결정나무 (DecisionTreeClassifier)모델의 
min_samples_split = 2 : 내부 노드를 분할하는 데 필요한 최소 샘플 수(기본 2개)
랜덤포레스트 (RandomForestClassifier)모델의 
min_samples_leaf=1 : leaf node를 만드는데 필요한 최소 샘플 수

'''

# 3.model tuning : GridSearch model
# 전체 데이터셋을 사용해, 최적 조건을 찾는다. 
from sklearn.model_selection import GridSearchCV # best parameters


parmas = {'n_estimators' : [100, 150, 200],
          'max_depth' : [None, 3, 5, 7],
          'max_features' : ["auto", "sqrt"],
          'min_samples_split' : [2, 10, 20],
          'min_samples_leaf' : [1, 10, 20]} # dict 정의 

grid_model = GridSearchCV(model, param_grid=parmas, 
                          scoring='accuracy',cv=5, n_jobs=-1)

grid_model.fit(X, y)

# 4. Best score & parameters 
print('best score =', grid_model.best_score_) 
# best score = 0.94214794181368
print('best parameters =', grid_model.best_params_)
'''
grid_model.best_params_)
best parameters = {'max_depth': None,
                   'max_features': 'sqrt', 
                   'min_samples_leaf': 1, 
                   'min_samples_split': 2,
                   'n_estimators': 150}
'''

# best_model 만들어서 최적의 조건인지 확인해보기 
best_model = RandomForestClassifier(max_depth= None,
                   max_features='sqrt', 
                   min_samples_leaf=1, 
                   min_samples_split= 2,
                   n_estimators=150)

# 위에서 홀드아웃으로 전체에서 비복원 추출한 데이터 사용 
best_model.fit(X= X_train, y = y_train)
best_model.score(X=X_test, y =y_test) # 0.9666666666666667
