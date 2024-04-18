# -*- coding: utf-8 -*-
"""
step02_logisticRegression_ROC.py

 - 로지스틱회귀모델 & ROC 평가 
"""

from sklearn.datasets import load_breast_cancer # dataset
from sklearn.linear_model import LogisticRegression # model ★★★
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 


################################
### 이항분류(binary class) 
################################

# 1. dataset loading 
X, y = load_breast_cancer(return_X_y=True)

print(X.shape) # (569, 30)
print(y) 


# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=1)


# 3. model 생성 
lr = LogisticRegression(solver='lbfgs', max_iter=100, random_state=1)  
'''
알고리즘과 값 조절에 따라 ROC 커브 출력은 달라진다. ★★★
solver='lbfgs', : 최적화에 사용되는 기본 알고리즘(solver) 
max_iter=100,  : 반복학습횟수 
random_state=None, : 난수 seed값 지정 
'''

model = lr.fit(X=X_train, y=y_train) 
dir(model)
'''
predict() : 'y클래스' 예측 
predict_proba() : y '확률' 예측
'''

# 4. model 평가 
# 기본적으로 y클래스를 예측한다 
y_pred = model.predict(X = X_test) # class 예측치 
y_pred_proba = model.predict_proba(X=X_test) # '확률'예측치 (ROC에서 사용)
y_true = y_test # 관측치 

# 1) 혼동행렬(confusion_matrix)
con_max = confusion_matrix(y_true, y_pred)
con_max
'''
[[ 59,   4],
[  5, 103]],
'''

# 2) 분류정확도 
acc = accuracy_score(y_true, y_pred)
print('accuracy =',acc) # accuracy = 0.9415204678362573
# (59+103)/len(con_max)


#############################
# ROC curve 시각화 
# (클래스 예측값 말고, 확률값이 필요하다)
#############################
'''
ROC y축에 TPR (민감도) 가 필요 =>  y_pred_proba[:, 1]   
'''
# 1) 확률 예측치
y_pred_proba = model.predict_proba(X = X_test) # 확률 예측 ★★★
# 그리고, 
# 0, 1 중 1(악성) 만 필요하다.
y_pred_proba = y_pred_proba[:, 1]   


# 2) ROC curve 
from sklearn.metrics import roc_curve # ★★★
import matplotlib.pyplot as plt 

fpr, tpr, _ = roc_curve(y_true, y_pred_proba) 


# fpr : x축 (1-특이도)
# tpr : y축에 들어감 
# _ : 임계값 등이 올수있는 데 필요없어서 반환받지 않는 값
plt.plot(fpr, tpr, color = 'red', label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()

'''
ROC curve FPR vs TPR  

ROC curve x축 : FPR(False Positive Rate) - 실제 음성을 양성으로 잘못 예측할 비율  
ROC curve y축 : TPR(True Positive Rate) - 실제 양성을 양성으로 정상 예측할 비율  
'''
