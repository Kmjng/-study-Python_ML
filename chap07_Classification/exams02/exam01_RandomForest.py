'''
 문) 당뇨병(diabetes.csv) 데이터 셋을 이용하여 다음과 같은 단계로 
     RandomForest 모델을 만드시오.

  <단계1> x, y 변수 선택 : x변수 : 1 ~ 8번째 칼럼, y변수 : 9번째 칼럼
  <단계2> 200개의 트리를 이용하여 모델 생성   
  <단계3> 비복원방식으로 300개 Test set 선정 
  <단계4> model 평가 : Test set 이용한 분류정확도  
  <단계5> 중요변수 시각화 & Top3 중요변수 찾기                   
'''

from sklearn.ensemble import RandomForestClassifier # 앙상블모델 
from sklearn.metrics import accuracy_score # 모델 평가   
import pandas as pd # csv file read 
import numpy as np # test set 선정 
import matplotlib.pyplot as plt # 중요변수 시각화 

# Dataset 가져오기    
dia = pd.read_csv('C:/ITWILL/4_Python_ML/data/diabetes.csv', 
                  header=None) # 제목 없음 

print(dia.info())

# 칼럼명 추가 
dia.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
               'Insulin','BMI','DiabetesPedigree','Age','Outcome']
print(dia.info()) 
'''
 0   Pregnancies       759 non-null    float64
 1   Glucose           759 non-null    float64
 2   BloodPressure     759 non-null    float64
 3   SkinThickness     759 non-null    float64
 4   Insulin           759 non-null    float64
 5   BMI               759 non-null    float64
 6   DiabetesPedigree  759 non-null    float64
 7   Age               759 non-null    float64
 8   Outcome           759 non-null    int64    => y 변수 
 (한글명 : 임신, 혈당, 혈압, 피부두께,인슐린,비만도지수,당뇨병유전,나이,결과)  
'''


# 단계1. X,y 변수 만들기  
X = dia.iloc[:, :-1]
y = dia.iloc[:, -1]

X.shape # (759, 8)
y.shape # (759,)  # 0 or 1로 구성되어 있음 
len(X) # 759


# 단계2. model 생성 : 200개 tree 학습 (Train) ★★★★
# 랜덤 복원추출
model = RandomForestClassifier(n_estimators = 200).fit(X,y) 

model.get_params() # model 파라미터 ★★★★
'''
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'sqrt',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 200,
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
'''

# 단계3. 비복원방식으로 Test set 300개 선정 
# 랜덤 비복원추출
idx = np.random.choice(a=len(X), size=300, replace = False)
len(idx) # 300 

X_test = X.iloc[idx] 
y_test = y.iloc[idx] # y_true
X_test

# 단계4. model 평가 : Test set 이용한 분류정확도
y_pred = model.predict(X= X_test) # 예측치 
print('accuracy =', accuracy_score(y_test, y_pred)) # 분류정확도 
# 1.0  # 여기서 사용된 테스트셋이 풀셋에 포함되어 있어서 1.0 나옴 
# model 평가 (2) - 교차검증 
from sklearn.model_selection import cross_validate
cv_test = cross_validate(model, X_test, y_test, cv = 5)
print(cv_test)
'''
{'fit_time': array([0.28176379, 0.26682472, 0.28377414, 0.26670527, 0.26654148]), 
 'score_time': array([0.01813102, 0.01644039, 0.01625109, 0.01677942, 0.0166235 ]), 
 'test_score': array([0.8       , 0.76666667, 0.66666667, 0.73333333, 0.71666667])}
'''

# 단계5. 중요변수 시각화 : Top3 변수 찾기  
x_names = list(dia.columns[:-1]) # x변수명 
x_size = len(x_names) # x변수 크기 

# 가로막대 차트 : 완성 
plt.barh(range(x_size), model.feature_importances_) 
plt.yticks(range(x_size), x_names) # y축 눈금 : x변수 
plt.xlabel('feature_importances')
plt.show()