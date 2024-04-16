######################################
### 결측치 처리
######################################

'''
- 특수문자를 결측치로 처리하는 방법 
'''

import pandas as pd 
pd.set_option('display.max_columns', 50) # 최대 50 칼럼수 지정

# 데이터셋 출처 : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data?select=data.csv
cancer = pd.read_csv(r'C:\ITWILL\4_Python_ML\data\breastCancer.csv')
cancer.info()
'''
RangeIndex: 699 entries, 0 to 698
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   id               699 non-null    int64  -> 제거 
 1   clump            699 non-null    int64 
 2   cell_size        699 non-null    int64 
 3   cell_shape       699 non-null    int64 
 4   adhesion         699 non-null    int64 
 5   epithlial        699 non-null    int64 
 6   bare_nuclei      699 non-null    object -> x변수(노출원자핵) : 숫자형 변환
 7   chromatin        699 non-null    int64 
 8   normal_nucleoli  699 non-null    int64 
 9   mitoses          699 non-null    int64 
 10  class            699 non-null    int64 -> y변수 
'''
print(df['class'].unique()) # >> [2 4]

# 1. 변수 제거 
df = cancer.drop(['id'], axis = 1) # 열축 기준 : id 칼럼 제거  


# 2. x변수 숫자형 변환 : object -> int형 변환  
df['bare_nuclei'] = df['bare_nuclei'].astype('int') # error 발생 



# 3. 특수문자 결측치 처리 & 자료형 변환 

# 1) 특수문자 결측치 대체   
import numpy as np 
df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan) 


# 2) 전체 칼럼 단위 결측치 확인 
# .any()
df.isnull().any() 


# 3) 결측치 제거  
new_df = df.dropna(subset=['bare_nuclei'])    
new_df.shape # (683, 10) : 16개 제거 


# 4) int형 변환 
new_df['bare_nuclei'] = new_df['bare_nuclei'].astype('int64') 


# (전처리)
# 4. y변수 레이블 인코딩 (전처리) : 10진수 변환 
from sklearn.preprocessing import LabelEncoder 
# 10진수로 인코딩하여 분류 

# 인코딩 객체 
encoder = LabelEncoder().fit(new_df['class']) # object 

# data변환 
labels = encoder.transform(new_df['class'])  
labels # 0 or 1 (범주가 2개라서 0,1이 끝임)

df['y']= labels
df.info()
 
