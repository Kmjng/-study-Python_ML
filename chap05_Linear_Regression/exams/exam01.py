'''  
문1) car_crashes 데이터셋을 이용하여 각 단계별로 다중선형회귀모델을 생성하시오.  
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sn # 데이터셋 로드 

# 미국의 51개 주 자동차 사고 관련 데이터셋 
car = sn.load_dataset('car_crashes')  
car.info()
'''
 0   total           51 non-null     float64 : 치명적 충돌사고 운전자 수 
 1   speeding        51 non-null     float64 : 과속 운전자 비율 
 2   alcohol         51 non-null     float64 : 음주 운전자 비율 
 3   not_distracted  51 non-null     float64 : 주시태만이 아닌 경우 충돌에 연루된 비율  
 4   no_previous     51 non-null     float64 : 이전 사고기록 없는 경우 충돌에 연루된 비율  
 5   ins_premium     51 non-null     float64 : 자동차보험료 
 6   ins_losses      51 non-null     float64 : 보험사가 입은 손해 
 7   abbrev          51 non-null     object : 주이름 
''' 


# 단계1 : abbrev 변수 제거하여 new_df 만들기  
new_df = car.drop('abbrev', axis = 1) 
new_df.shape # (51, 7)

# 단계2 : total과 비교하여 상관계수가 0.2미만의 모든 변수 제거 후 new_df에 반영  
corr = new_df.corr()
type(corr) # pandas.core.frame.DataFrame
corr
'''
                   total  speeding  ...  ins_premium  ins_losses
total           1.000000  0.611548  ...    -0.199702   -0.036011
speeding        0.611548  1.000000  ...    -0.077675   -0.065928
alcohol         0.852613  0.669719  ...    -0.170612   -0.112547
not_distracted  0.827560  0.588010  ...    -0.174856   -0.075970
no_previous     0.956179  0.571976  ...    -0.156895   -0.006359
ins_premium    -0.199702 -0.077675  ...     1.000000    0.623116
ins_losses     -0.036011 -0.065928  ...     0.623116    1.000000

[7 rows x 7 columns]
'''
corr_tot = corr.loc['total']
corr_tot_1 = corr.loc['total'] >= 0.2 # 시리즈 
corr_tot
corr_tot.shape #( 7,)
type(corr_tot) # Series
'''
total             1.000000
speeding          0.611548
alcohol           0.852613
not_distracted    0.827560
no_previous       0.956179
ins_premium      -0.199702
ins_losses       -0.036011
Name: total, dtype: float64
'''

print(corr_tot.values)
# ★★★
# 시리즈라 index, value로 구성되어있지만, 
# for 문 적용할 때, iterrows()말고 dict처럼 .items() 사용한다.
cols = []
for idx, val in corr_tot.items():
    if val >=0.2 or val <= -0.2: 
        cols.append(idx)

new_df = new_df[cols]
new_df.columns 
''' ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous'] '''

new_df
'''
    total  speeding  alcohol  not_distracted  no_previous
0    18.8     7.332    5.640          18.048       15.040
1    18.1     7.421    4.525          16.290       17.014
2    18.6     6.510    5.208          15.624       17.856
...
9    17.9     3.759    5.191          16.468       16.826
10   15.6     2.964    3.900          14.820       14.508
11   17.5     9.450    7.1
'''
# 단계3 : new_df에서 종속변수는 total, 나머지 변수는 독립변수  
X = new_df.iloc[:,1:] 
y = new_df.iloc[:,0] # 종속변수 
X.shape # (51, 4)
y.shape # (51,)

# 단계4 : train/test split(70% vs 30%)
X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.3)


# 단계5. 회귀모델 생성 : train set 이용 
car_model = LinearRegression().fit(X=X_train, y=y_train) 

# 단계6. 모델 평가 : test set 이용  
y_pred = car_model.predict(X_test) # 예측치 
y_true = y_test # 관측치(정답)
 
# 1) MSE
mse =  mean_squared_error(y_true, y_pred)
print('MSE =', mse)  # MSE = 1.318899854148123

# 2) 결정계수 
score = r2_score(y_true, y_pred)
print('결정계수 = %.5f'%score)
# 결정계수 = 0.87898
