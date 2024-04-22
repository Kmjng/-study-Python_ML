"""
기사 실기시험 유형 

문3) 다음 타이타닉(titanic) 데이터셋으로 로지스틱회귀모델을 만들고, 모델을 평가하시오.
     <작업절차>
      1. titanic_train.csv의 자료를 전처리한 후 75%을 이용하여 모델을 학습하고,
         25%로 학습된 모델을 검증한다.(검증 대상 : 모델 성능, 과적합 여부)
      2. 검증된 모델에 titanic_test.csv의 X변수를 적용하여 예측치를 구한다.
      
    훈련셋/검증셋 : titanic_train.csv(X, y변수 포함)  
    평가셋 : titanic_test.csv(X변수만 포함) 
"""

import pandas as pd
pd.set_option('display.max_columns',15) # 최대 15개 칼럼 출력 

from sklearn.linear_model import LogisticRegression # class - model
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn.model_selection import train_test_split # split


### 1단계 : train.csv 가져오기 
train = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\titanic_train.csv")
train.info()
'''
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 0   PassengerId  891 non-null    int64   -> 제거 
 1   Survived     891 non-null    int64   -> y변수 
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object  -> 제거
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object  -> 제거
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object  
 11  Embarked     889 non-null    object
'''


### 2단계 : 불필요한 변수('PassengerId','Name','Ticket') 제거 후 new_df 만들기 
new_df = train.drop(['PassengerId','Name','Ticket'], axis = 1)
train.shape # (891, 12)
new_df.shape # (891, 9)

### 3단계 : 결측치 확인 및 처리 

# 1) 결측치(NaN) 확인
new_df.isnull().sum()
'''
Survived      0
Pclass        0
Sex           0
Age         177
SibSp         0
Parch         0
Fare          0
Cabin       687
Embarked      2
dtype: int64
'''
null_series = new_df.isnull().sum() 
null_column = [i for i,j in null_series.items() if j > 500 ]

# 2) 결측치가 50% 이상 칼럼 제거 후 new_df에 반영     
new_df.index.shape # (891)
new_df.index.shape[0] 
new_df = new_df.drop(null_column, axis =1)
new_df.columns 
''' ['Survived', 'Pclass', 'Sex', 'Age', 
'SibSp', 'Parch', 'Fare','Embarked'] '''
# 3) Age 칼럼의 결측치를 평균으로 대체 후 new_df에 반영 
new_df.Age = new_df.Age.fillna(new_df.Age.mean())
new_df.Age.isnull().sum() # 0 

# 4) Embarked 칼럼의 결측치를 가장 많이 출현한 값으로 대체 후 new_df 적용
new_df.Embarked = new_df.Embarked.fillna(new_df.Embarked.mode())

### 단계4 : X변수, y변수 선정   

# 1) X변수 만들기 : new_df에서 'Survived' 칼럼을 제외한 나머지 칼럼 이용  
new_df.columns
X = new_df[new_df.columns[1:]] # 독립변수
X.columns
# 2) y변수 만들기  : new_df에서 'Survived' 칼럼 이용   
# ★★★★
y = new_df[new_df.columns[0]] # 종속변수

### 단계5 : X변수 전처리     

# 1) object형 변수를 대상으로 k-1개 가변수 만들기
# Sex, Embarked 
import pandas as pd
X_d = pd.get_dummies(X, columns=['Sex','Embarked'], drop_first =True, dtype='uint8')
X_d.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 8 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Pclass      891 non-null    int64  
 1   Age         891 non-null    float64
 2   SibSp       891 non-null    int64  
 3   Parch       891 non-null    int64  
 4   Fare        891 non-null    float64
 5   Sex_male    891 non-null    uint8  
 6   Embarked_Q  891 non-null    uint8  
 7   Embarked_S  891 non-null    uint8  
dtypes: float64(2), int64(3), uint8(3)
memory usage: 37.5 KB
'''
# 2) X변수를 대상으로 최소-최대 정규화 후 X_train적용 
X_scaled = StandardScaler().fit_transform(X=X_d)
X_scaled.shape # (891, 8)

### 단계6 : 훈련셋(train)/검증셋(val) split(75% : 25%) 
X = pd.DataFrame(X_scaled, columns = X_d.columns) # 데이터프레임으로 만들기 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25,random_state=123 )

### 단계7 : 로지스틱회귀모델 생성 & 결정계수 model 평가 

# 1) 회귀모델 생성 : 훈련셋 이용(X_train, y_train)
model = LogisticRegression(solver='lbfgs',multi_class='auto',
                           max_iter=100, random_state = 1).fit(X_train, y_train)
y_pred = model.predict(X_val)
y_true = y_val

# 2) 회귀모델 검증 : 검증셋 이용(X_val, y_val) 
train_score = model.score(X_train, y_train) 
train_score # 0.7979041916167665

val_score = model.score(X=X_val, y=y_val)
val_score # 0.7982062780269058


#######################################################
### 8단계 테스트셋(titanic_test.csv)으로 model 평가 
# 위에서 만든 model을 적용해 예측한다!!
test = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\titanic_test.csv")
test.info() # y변수 없음 
'''
 0   PassengerId  418 non-null    int64   - 제거 
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object  - 제거 
 3   Sex          418 non-null    object  - 더미변수 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object  - 제거 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object  - 제거 
 10  Embarked     418 non-null    object  - 더미변수 
'''
 
### 1. 불필요한 변수 제거 new_test 만들기 
new_test = test.drop(['PassengerId','Name','Ticket','Cabin'], 
                     axis =1)
print(new_test.info())
'''
 0   Pclass    418 non-null    int64  
 1   Sex       418 non-null    object 
 2   Age       332 non-null    float64
 3   SibSp     418 non-null    int64  
 4   Parch     418 non-null    int64  
 5   Fare      417 non-null    float64
 6   Embarked  418 non-null    object
''' 

### 2. 결측치 확인 및 평균 대체  

# 1) 결측치(NaN) 확인
new_test.isnull().any()
new_test.isnull().sum()
'''
Age         86
Fare         1
'''

# 2) Age, Fare 칼럼의 결측치를 평균으로 대체 후 new_df에 반영 
new_test['Age'].fillna(new_test['Age'].mean(), inplace=True)
new_test.isnull().any()

new_test['Fare'].fillna(new_test['Fare'].mean(), inplace=True)
new_test.isnull().any()


### 3. X변수 전처리     

# 1) 문자형 변수를 대상으로 k개 가변수 만들기(원핫 인코딩)
X_test = pd.get_dummies(new_test, columns = ['Sex', 'Embarked'], drop_first= True)
X_test.info()
'''
 0   Pclass      418 non-null    int64  
 1   Age         418 non-null    float64
 2   SibSp       418 non-null    int64  
 3   Parch       418 non-null    int64  
 4   Fare        418 non-null    float64
 5   Sex_male    418 non-null    uint8  
 6   Embarked_Q  418 non-null    uint8  
 7   Embarked_S  418 non-null    uint8
''' 

# 2) X변수를 대상으로 표준화 후 X_train적용 
X_scaled = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns = X_test.columns)

X_test.info()
'''
RangeIndex: 418 entries, 0 to 417
Data columns (total 8 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Pclass      418 non-null    int64  
 1   Age         418 non-null    float64
 2   SibSp       418 non-null    int64  
 3   Parch       418 non-null    int64  
 4   Fare        418 non-null    float64
 5   Sex_male    418 non-null    uint8  
 6   Embarked_Q  418 non-null    uint8  
 7   Embarked_S  418 non-null    uint8 
''' 

### 4. 학습된 모델(model) X변수를 적용하여 예측치 만들기 & 출력 
y_pred = model.predict(X_test) # class 예측 
print(y_pred)
