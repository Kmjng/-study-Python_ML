######################################
### 3. 데이터 인코딩 
######################################

"""
데이터 인코딩 : 
    머신러닝 모델에서 범주형변수를 대상으로 숫자형의 목록으로 변환해주는 전처리 작업
 - 방법 1   
 - 레이블 인코딩(label encoding) : 트리계열모형(의사결정트리, 앙상블)의 변수 대상(10진수) 
                                    ★★ 이외의 회귀계열 모형에서는 가중치로 인식되기 때문에
                                        사용되지 않음.
 - 방법 2  
 - 원-핫 인코딩(one-hot encoding) : 
     회귀계열모형(선형,로지스틱,SVM,신경망)의 변수 대상(2진수) 
   -> 회귀모형에서는 인코딩값이 가중치로 적용되므로 원-핫 인코딩으로 변환 
   분류할 범주가 k개라면, 변수는 k개가 아니라 (k-1)개의 가변수를 이용한다. 
- k개의 변수
1 0 0 0 - A 
0 1 0 0 - AB
0 0 1 0 - B
0 0 0 1 - O
- k-1개의 변수
A, AB, B, O
0 0 0 - base(A) 
1 0 0 - AB
0 1 0 - B
0 0 1 - O

"""


import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\4_Python_ML\data\skin.csv")
data.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외★ (구분자)
 1   gender       30 non-null     object -> x변수 
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


## 1. 변수 제거 : cust_no
df = data.drop('cust_no', axis = 1) # axis=1 : 칼럼제거
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   gender  30 non-null     object
 1   age     30 non-null     int64 
 2   job     30 non-null     object
 3   marry   30 non-null     object
 4   car     30 non-null     object
 5   label   30 non-null     int32 
dtypes: int32(1), int64(1), object(4)
memory usage: 1.4+ KB
'''


### 2. 레이블 인코딩 : 트리모델 계열의 x, y변수 인코딩  
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 

# 1) 쿠폰 반응 범주 확인 
df.cupon_react.unique() # array(['NO', 'YES'], dtype=object) 
# 2개로 분류

# 2) 인코딩
encoder = LabelEncoder() # encoder 객체 
label = encoder.fit_transform(df['cupon_react']) # data 반영 
label # 0,1 

# 3) 칼럼 추가 
df['label'] = label
df= df.drop('cupon_react', axis =1)
df.info()


### 3. 원-핫 인코딩 : 회귀모델 계열의 x변수 인코딩  

# 1) k개 목록으로 가변수(더미변수) 만들기 
# get_dummies() 
# df 변수들: gender, age, job, marry, car, label
df_dummy = pd.get_dummies(data=df) # 기준변수 포함 
df_dummy # age  gender_female  gender_male  ...  marry_YES  car_NO  car_YES -> (1+8)


# 2) 특정 변수 선택   
df_dummy2 = pd.get_dummies(data=df, columns=['label','gender','job','marry'])
df_dummy2.info() # Data columns (total 12 columns):
# 전체 변수는 10개 (age,car, 4개*2)
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 10 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            30 non-null     int64 
 1   car            30 non-null     object
 2   label_0        30 non-null     bool  
 3   label_1        30 non-null     bool  
 4   gender_female  30 non-null     bool  
 5   gender_male    30 non-null     bool  
 6   job_NO         30 non-null     bool  
 7   job_YES        30 non-null     bool  
 8   marry_NO       30 non-null     bool  
 9   marry_YES      30 non-null     bool  
dtypes: bool(8), int64(1), object(1)
memory usage: 852.0+ bytes
'''


# 3) k-1개 목록으로 가변수(더미변수) 만들기   
# 기준변수 제외(권장)
df_dummy3 = pd.get_dummies(data=df, drop_first=True, dtype='uint8') 
df_dummy3  
'''
    age  label  gender_male  job_YES  marry_YES  car_YES
0    30      0            1        0          1        0
1    20      0            0        1          1        1
2    20      0            0        1          1        0
3    40      0            0        0          0        0
...
26   30      1            0        1          1        1
27   40      0            0        1          0        1
28   40      1            1        1          1        0
29   40      1            0        1          1        0
'''
df_dummy3.info()
# 전체 변수는 6개 (age,car, 4개*1) 
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   age          30 non-null     int64
 1   label        30 non-null     int32
 2   gender_male  30 non-null     bool # 기준변수: gender_female (0)
 3   job_YES      30 non-null     bool 
 4   marry_YES    30 non-null     bool 
 5   car_YES      30 non-null     bool 
dtypes: bool(4), int32(1), int64(1)
memory usage: 612.0 bytes
'''
# 밑의 내용을 참고하여 male을 기준변수로 바꾸기 
# 1) category로 형 변환 (순서바꾸기 위해)
df['gender']= df['gender'].astype('category')
# 2) 카테고리타입 활용해서 순서바꾸기 (cat.set_categories(['범주1','범주2',..]))
df['gender'] = df['gender'].cat.set_categories(['male','female'])
# 3) 더미변수 (k-1개) 생성 
df_dummy4 = pd.get_dummies(data=df, columns=['gender'], drop_first=True)
df_dummy4.info()


###############################
## 가변수 기준(base) 변경하기  
###############################
import seaborn as sn 
iris = sn.load_dataset('iris')
iris['species'].unique() # ['setosa', 'versicolor', 'virginica']


# 1. 가변수(dummy) : k-1개 
iris_dummy = pd.get_dummies(data = iris, columns=['species'], drop_first=True)
# drop_first=True : 첫번째 범주 제외(기준변수)
# 변수 하나에 대해 더미변수 생성 (3-1 개)
# 변수 갯수 : ...., versicolor, virginica

# ★★★★
# 2. base 기준 변경 : 범주 순서변경('virginica' -> 'versicolor' -> 'setosa') 
# object -> category 변환 
iris['species'] = iris['species'].astype('category')
iris['species'] = iris['species'].cat.set_categories(['virginica','versicolor','setosa'])


# 3. 가변수(dummy) : k-1개 
iris_dummy2 = pd.get_dummies(data=iris, columns=['species'], drop_first=True)



