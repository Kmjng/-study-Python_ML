# -*- coding: utf-8 -*-
"""
RandomForest 앙상블 모델 
"""

from sklearn.ensemble import RandomForestClassifier # 분류 model
from sklearn.model_selection import train_test_split # dataset split  
from sklearn.datasets import load_wine # dataset 

# 평가 도구 
from sklearn.metrics import confusion_matrix, classification_report

# 1. dataset load
wine = load_wine()

X, y = wine.data, wine.target
X.shape  # (178, 13) 


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123) # 90% 훈련셋 


# 3. model 생성 
'''
RandomForestClassifier의 
주요 hyper parameter(default)
 n_estimators=100 : tree 개수 ★★★★ 단일 트리와의 차이점 
 criterion='gini' : 중요변수 선정 기준 
 max_depth=None : 트리 깊이 
 min_samples_split=2 : 내부 노드 분할에 필요한 최소 샘플 수
'''

# 훈련셋 적용 : 풀셋 적용 ★★ (이렇게 하면 정확도 1.0 으로 나옴 )
# 랜덤포레스트 : a개의 랜덤복원추출 
model = RandomForestClassifier(random_state=123).fit(X = X, y = y) 



# 4. model 평가 : 테스트셋 평가 ★★★★
y_pred = model.predict(X = X_test)


# 혼동행렬
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
'''
[[3 0 0]
 [0 6 0]
 [0 0 9]]
'''


# 분류 리포트 
print(classification_report(y_test, y_pred))
# 다 맞힘 (당연)

# 5. 중요변수 시각화 
print('중요도 : ', model.feature_importances_)
len(model.feature_importances_) # 13개 칼럼의 중요도 
'''
중요도 :  [0.10968347 0.0327231  0.0122728  0.02942002 0.02879057 0.05790394
 0.13386676 0.00822129 0.02703471 0.158535   0.07640667 0.13842681
 0.18671486]
'''
x_names = wine.feature_names # x변수 이름  
x_size = len(x_names) # x변수 개수; 13

import matplotlib.pyplot as plt 

# 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) # (y, x) 
plt.yticks(range(x_size), x_names)   
plt.xlabel('feature_importances') 
plt.show()


# Top 5 중요도 변수 선택 
imp = model.feature_importances_
Top_5_idx = imp.argsort()[:5:-1] 
print(Top_5_idx)
Top_5 = [x_names[idx] for idx in Top_5_idx]
Top_5
'''
['proline',
 'color_intensity',
 'od280/od315_of_diluted_wines',
 'flavanoids',
 'alcohol',
 'hue',
 'total_phenols']
'''