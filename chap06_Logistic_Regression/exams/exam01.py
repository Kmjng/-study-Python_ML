# -*- coding: utf-8 -*-
"""
문1) 주어진 자료를 대상으로 조건에 맞게 단계별로 로지스틱 회귀모델(이항분류)를 생성하시오.  
    조건1> cust_no, cor 변수 제거 
    조건2> object형 X변수 : OneHotEncoding(k-1개)
    조건3> object형 y변수 : LabelEncoding 
    조건4> 모델 평가 : 혼동행렬과 분류정확도
    조건5> ROC curve 시각화  
"""

from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 
from sklearn.preprocessing import LabelEncoder # 레이블 인코딩 도구 

import pandas as pd # pd.get_dummies() : 원-핫 인코딩 도구 

df = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\skin.csv")
df.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외 
 1   gender       30 non-null     object -> x변수(성별) 
 2   age          30 non-null     int64  -> x변수(나이)
 3   job          30 non-null     object -> x변수(직업유무)
 4   marry        30 non-null     object -> x변수(결혼여부)
 5   car          30 non-null     object -> 변수 제외 
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


# 단계1. 변수 제거 : cust_no, car
new_df = df.drop(['cust_no','car', 'cupon_react'], axis = 1)


# 단계2. object형 변수 인코딩  
# 1) X변수 OneHotEncoding : k-1개 가변수 만들기 gender, job, marry 변수
X = pd.get_dummies(data=new_df, columns=['gender','job','marry'] ,drop_first=True, dtype='uint8')
X
'''
    age cupon_react(지움) gender_male  job_YES  marry_YES
0    30          NO            1        0          1
1    20          NO            0        1          1
2    20          NO            0        1          1
3    40          NO            0        0          0
...
27   40          NO            0        1          0
28   40         YES            1        1          1
29   40         YES            0        1          1
'''
# 2) y변수 LabelEncoding : cupon_react 변수
y = LabelEncoder().fit_transform(df['cupon_react'])
X.shape 

# 단계3. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=34)


# 단계4. model 생성 
lgr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   max_iter=100, 
                   multi_class='auto')
model= lgr.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_true = y_test
# 단계5. model 평가 

# 1) 혼동행렬 
con_m = confusion_matrix(y_true, y_pred)
print(con_m)
'''
[[4 0]
 [1 4]]
'''
# 2) 분류정확도 
acc =accuracy_score(y_true, y_pred)
acc # 0.8888888888888888


# 단계6. ROC curve 시각화
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

y_pred_proba = model.predict_proba(X_test)[:,1]
# y_pred_proba[:,1]

fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
asdf = pd.DataFrame({'fpr':fpr, 'tpr':tpr})
asdf
fdsa =pd.DataFrame({'y_true':y_true, 'y_pred_proba':y_pred_proba})
fdsa

'''
    fpr  tpr
0  0.00  0.0
1  0.00  0.4
2  0.00  0.8
3  0.75  0.8
4  0.75  1.0
5  1.00  1.0  (임계값 갯수만큼 그래프에 찍힘)
임계값 : 모델이 내부적으로 계산해 수행
'''
plt.plot(fpr, tpr, color ='red',label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()