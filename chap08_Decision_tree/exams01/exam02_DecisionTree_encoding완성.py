# -*- coding: utf-8 -*-
'''
 문2) 다음 데이터 셋을 이용하여 단계별로 Decision Tree 모델을 생성하시오.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report # model evaluation

path = r'C:/ITWILL/4_Python_ML/data'
data = pd.read_csv(path +'/dataset.csv') 
data.info()
'''
RangeIndex: 217 entries, 0 to 216
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   resident  217 non-null    int64    : x변수 
 1   gender    217 non-null    int64    : x변수 
 2   job       205 non-null    float64  : x변수 
 3   age       217 non-null    int64    : x변수 
 4   position  208 non-null    float64  : y변수 
 5   price     217 non-null    float64  : x변수 
 6   survey    217 non-null    int64    : x변수  
'''
 
print(data)
'''
     resident  gender  job  age  position  price  survey
0           1       1  1.0   46       4.0    4.1       1
1           2       1  2.0   54       1.0    4.2       2
2           4       2  NaN   45       2.0    3.5       2
3           5       1  3.0   62       1.0    5.0       1
4           3       1  2.0   57       NaN    5.4       2
'''
data.isnull().sum()
'''
resident     0
gender       0
job         12
age          0
position     9
price        0
survey       0
'''

# 단계1 : 결측치를 포함한 모든 행 제거 후 new_data 생성 
new_data = data.dropna(axis=0, subset=['job','position'])
new_data.shape  #  (198, 7)


# 단계2 : resident, gender, job, position 변수 레이블인코딩
 
from sklearn.preprocessing import LabelEncoder # 인코딩 도구

new_data['resident'] = LabelEncoder().fit_transform(new_data['resident'])
new_data['gender'] = LabelEncoder().fit_transform(new_data['gender'])
new_data['job'] = LabelEncoder().fit_transform(new_data['job'])
new_data['position'] = LabelEncoder().fit_transform(new_data['position'])

new_data.info()
'''
 0   resident  198 non-null    int64  
 1   gender    198 non-null    int64  
 2   job       198 non-null    int64  
 3   age       198 non-null    int64  
 4   position  198 non-null    int64  
 5   price     198 non-null    float64
 6   survey    198 non-null    int64 
'''

# 단계3 : X, y변수 선택 
X = new_data[['resident','gender','job','age','price','survey']]
y = new_data['position']



# 단계4 : 훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 123)

print(X_train.shape)   # (148, 6)
print(y_train.shape)   # (148,)


# 단계5 : 기본모델 생성 
model = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)


# 단계6. 모델 평가 : 분류정확도, 혼동행렬  
y_pred= model.predict(X_test) # 예측치

# 혼동 행렬
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)     
'''
[[16  0  0  0  3]
 [ 0  9  0  0  1]
 [ 0  0  7  0  0]
 [ 0  1  0  5  0]
 [ 0  2  0  0  6]]
'''

classification_report = classification_report(y_test, y_pred)
print(classification_report)
'''
              precision    recall  f1-score   support

           0       1.00      0.84      0.91        19
           1       0.75      0.90      0.82        10
           2       1.00      1.00      1.00         7
           3       1.00      0.83      0.91         6
           4       0.60      0.75      0.67         8

    accuracy                           0.86        50
   macro avg       0.87      0.87      0.86        50
weighted avg       0.89      0.86      0.87        50
'''

# 그리드서치 활용 ★★★ 최적의 파라미터 찾기 
# 단계7 : best parameter 찾기 : 

from sklearn.model_selection import GridSearchCV # best parameters

parmas = {'criterion' : ['gini', 'entropy'], # 중요변수 선택 
          'max_depth' : [None, 3, 4, 5, 6],  # 트리 깊이 
          'min_samples_split': [2, 3, 4]}  # 내부노드 분할 최소 샘플 수 



# 1) 기본 model을 대상으로 5겹 교차검정으로 수행
grid_model = GridSearchCV(model, param_grid=parmas, 
                   scoring='accuracy',cv=5, n_jobs=-1).fit(X, y)


# 2) Best score 확인 
print('best score =', grid_model.best_score_)
#best score = 0.9294871794871795

# 3) Best parameters 확인  
print('best parameters =', grid_model.best_params_)
'''
best parameters = {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 2}
'''

# 4) best parameters 적용 : new model 생성 & 평가  
obj = DecisionTreeClassifier(criterion='gini', max_depth= 4, 
                             min_samples_split = 2)

new_model = obj.fit(X=X_train, y=y_train) 

train_score = new_model.score(X=X_train, y=y_train)
print(train_score) # 0.9662162162162162

test_score = new_model.score(X=X_test, y=y_test)
print(test_score) # 0.9
