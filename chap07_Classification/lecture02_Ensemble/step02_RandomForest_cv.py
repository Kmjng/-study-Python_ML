'''
k겹 교차검정(cross validation)
 - 전체 dataset을 k등분 
 - 검정셋과 훈련셋을 서로 교차하여 검정하는 방식 
'''

from sklearn.datasets import load_digits # 0~9 손글씨 이미지 
from sklearn.ensemble import RandomForestClassifier # RM model 
from sklearn.metrics import accuracy_score # 평가 
from sklearn.model_selection import cross_validate # 교차검정 ★★
from sklearn.model_selection import train_test_split # 홀드아웃 ★★
import numpy as np # Testset 선정

# 1. dataset load 
digits = load_digits()

X = digits.data
y = digits.target

X.shape # (1797, 64) 
y.shape # (1797,)
len(X) # 1797
# 2. model 생성 : tree 100개 학습 
model = RandomForestClassifier().fit(X, y) # full dataset 이용 


# 3. Test set 선정 : 500 개의 이미지 ★★★★
# 랜덤 추출 0~ 499 ; np.random.choice() 
'''
a=len(X) ; X 배열의 길이를 지정합니다. 추출할 샘플의 범위로 사용
size=500 ; 추출할 샘플의 개수를 지정
replace = False ; 복원추출 x  
'''  

idx = np.random.choice(a=len(X), size=500, replace = False)
type(idx) # 배열 
X_test = X[idx]
y_test = y[idx]


# 4. 평가셋 이용 : model 평가(1회) 
y_pred = model.predict(X = X_test) # 예측치  
y_pred 

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # 1.0 # 풀셋 사용해서 1.0 나옴 


# 5. k-fold 교차검정 이용 : model 평가(5회; 5개로 쪼갬) ★★★
# 비복원 추출 방법 
score = cross_validate(model, X_test, y_test, cv=5)
print(score)
'''

{'fit_time': array([0.18794107, 0.18745613, 0.19630933, 
                    0.19335079, 0.19078588]), 
 'score_time': array([0.01556945, 0.        , 0.        , 0.01562166, 0.00106478]), 
 'test_score': array([0.93, 0.97, 0.95, 0.96, 0.93])}
여기서 test_score 가 '분류 정확도'

'''
# 산술평균으로 model 성능 결정 
score['test_score'].mean() # 0.9479999999999998
