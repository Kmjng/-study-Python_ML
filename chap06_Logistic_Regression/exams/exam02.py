'''
문2) wine 데이터셋을 이용하여 조건에 맞게 단계별로 로지스틱회귀모델(다항분류)을 생성하시오. 
  조건1> train/test - 70:30비율
  조건2> y 변수 : wine.target 
  조건3> x 변수 : wine.data
  조건4> 모델 평가 : confusion_matrix, 분류정확도[accuracy]
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 단계1. wine 데이터셋 로드 
wine = load_wine()
# 와인 분류(0,1,2으로 인코딩되어 있음)
# x변수들은 연속형 데이터
wine.DESCR

# 단계2. x, y변수 선택 
wine_x = wine.data # x변수 
wine_y = wine.target # y변수
wine_x.shape # (178,3)
wine_y.shape # (178,) 
wine.feature_names
'''
['alcohol',
 'malic_acid',
 'ash',
 'alcalinity_of_ash',
 'magnesium',
 'total_phenols',
 'flavanoids',...
'''
wine.target_names # ['class_0', 'class_1', 'class_2']


# 단계3. train/test split(70:30)
X_train, X_test, y_train, y_test = train_test_split(wine_x, wine_y, 
                                                   test_size = 0.3, random_state=1 )

# 단계4. model 생성  : solver='lbfgs', multi_class='multinomial'
model = LogisticRegression(solver='lbfgs',multi_class='multinomial',
                           max_iter=100, random_state = 1).fit(X_train, y_train)

y_pred = model.predict(X_test)
y_true = y_test
# 단계5. 모델 평가 : accuracy, confusion matrix
acc = metrics.accuracy_score(y_true, y_pred)
acc # 0.9629629629629629
con_mat = metrics.confusion_matrix(y_true, y_pred)
con_mat
'''
[[22,  1,  0],
[ 0, 19,  0],
[ 0,  1, 11]]
'''
# 단계6. test셋으로 확률 예측하여 class가 2인 관측치만 출력(예시 참고) 
import pandas as pd
y_proba = model.predict_proba(X_test)[:,2]
class_df = pd.DataFrame({'y정답':y_true, 'y예측치 확률':y_proba})
class_df[class_df['y정답']==2]
'''
2로 분류된 관측치들의 '분류2'에 대한 확률
    y정답  y예측치
0     2  0.938881
2     2  0.959511
5     2  0.999500
...
41    2  0.950416
51    2  0.605581
52    2  0.993588
'''