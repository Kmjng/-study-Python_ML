'''
- XGBoost 앙상블 모델 테스트
- Anaconda Prompt에서 패키지 설치 
  pip install xgboost
'''

from xgboost import XGBClassifier # model ★★
from xgboost import plot_importance # 중요변수(x) 시각화  
from sklearn.datasets import make_blobs # 클러스터 생성 dataset ★★
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


dir(XGBClassifier)

# 1. 데이터셋 로드 : blobs
X, y = make_blobs(n_samples=2000, n_features=4, centers=3, 
                   cluster_std=2.5, random_state=123)
'''
XGBClassifier의 Hyperparameter 
n_samples : 표본크기 
n_features : x 변수 갯수 
centers : 클래스 갯수 
        centers = 2 : 이항분류 
        centers = n : 다항분류
cluster_std : 클러스터 표준편차 
            클러스터 내의 데이터 포인트가 중심에서 얼마나 퍼져 있는지
            0에 근사하면 복잡도 감소 ( 이거 확인해보기 ★★★★★)
        =>  XGBClassifier의 클러스터링을 사용하는 경우에만 해당
'''


X.shape # (2000, 4)
y # array([1, 1, 0, ..., 0, 0, 2]) # 0,1,2 세 개로 구성됨

# blobs 데이터 분포 시각화 
plt.title("three cluster dataset")
plt.scatter(X[:, 0], X[:, 1], s=100, c=y,  marker='o') # color = y범주
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. 훈련/검정 데이터셋 생성 (홀드아웃)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)



# 3. XGBOOST model 
model = XGBClassifier(objective='multi:softprob') # objective : 활성함수 
# softprob : 소프트맥스 
'''
centers = 2 이면 이항분류라, 시그모이드(로지스틱) 활성함수 
binary-class classification : objective='binary:logistic' 

centers = 3 이상이면 다항분류라, 소프트맥스 활성함수를 사용한다. 
multi-class classification : objective='multi:softprob' 
'''


# train data 이용 model 생성 : 트리 갯수 default 는 100개
eval_set = [(X_test, y_test)] # 평가셋 (안에 튜플)
model.fit(X_train, y_train, eval_set = eval_set, eval_metric='merror') 
# 학습과 검증을 동시에 시행함 ★★★★
'''
[0]	validation_0-merror:0.09167
[1]	validation_0-merror:0.09500
[2]	validation_0-merror:0.10167
...
[97]	validation_0-merror:0.10167
[98]	validation_0-merror:0.10000
[99]	validation_0-merror:0.10167
'''
# 오차가 점점 감소하는 것을 확인할 수 있음.  ????

'''
eval_metric : 학습과정에서 평가방법 
binary-class classification 평가방법 : eval_metric = 'error'
multi-class classification 평가방법 : eval_metric = 'merror'

'''
print(model) 



# 4. model 평가 
y_pred = model.predict(X_test) 
acc = accuracy_score(y_test, y_pred)
print('분류정확도 =', acc)
# 분류정확도 = 0.9166666666666666
# (이진분류로 만들면) 분류정확도 = 0.998333333

report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.87      0.91      0.89       206  # 클래스마다 균등비율로
           1       0.99      0.99      0.99       196
           2       0.89      0.85      0.87       198

    accuracy                           0.92       600
   macro avg       0.92      0.92      0.92       600
weighted avg       0.92      0.92      0.92       600
'''

# 5. fscore 중요변수 시각화  
fscore = model.get_booster().get_fscore()
print("fscore:",fscore) 
'''
fscore: {'f0': 844.0, 'f1': 772.0, 'f2': 1058.0, 'f3': 738.0}
세 번째 변수 f2의 중요도가 제일 큼
'''

# 중요변수 시각화
plot_importance(model) 
plt.show()
