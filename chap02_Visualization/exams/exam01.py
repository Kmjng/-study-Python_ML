'''
문1) dataset.csv 파일을 이용하여 다음과 같은 단계로 교차테이블과 누적막대차트를 그리시오.
  <단계1> 교차테이블 결과를 대상으로 만족도 1,3,5만 선택하여  subset 만들기   
  <단계2> 생성된 데이터프레임 대상 칼럼명 수정 : ['seoul','incheon','busan']
  <단계3> 생성된 데이터프레임 대상  index 수정 : ['male', 'female']     
  <단계4> 생성된 데이터프레임 대상 누적가로막대차트 그리기
'''

import pandas as pd
import matplotlib.pyplot as plt

path = r'C:\ITWILL\4_Python_ML\data'
dataset = pd.read_csv(path + '/dataset.csv')
dataset.info()
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 217 entries, 0 to 216
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   resident  217 non-null    int64  
 1   gender    217 non-null    int64  # 명목척도
 2   job       205 non-null    float64
 3   age       217 non-null    int64  
 4   position  208 non-null    float64
 5   price     217 non-null    float64
 6   survey    217 non-null    int64  # 등간척도
dtypes: float64(3), int64(4)
memory usage: 12.0 KB
'''
dataset
'''     resident  gender  job  age  position  price  survey
0           1       1  1.0   46       4.0    4.1       1
1           2       1  2.0   54       1.0    4.2       2
...
214         3       1  3.0   24       5.0    3.5       2
215         4       1  3.0   59       1.0    5.5       2
216         1       1  3.0   27       4.0    2.4       2

[217 rows x 7 columns]
'''
# 성별(gender)과 만족도(survey) 칼럼으로 교차테이블  작성 
table =  pd.crosstab(index=dataset['gender'], columns=dataset['survey'])
table
'''
survey   1   2   3   4  5
gender                   
1       10  51  44  13  5
2        4  36  42  11  1
'''

# <단계1> 교차테이블 결과를 대상으로 만족도 1,3,5만 선택하여 subset 만들기    
table_sub = table.iloc[:,0::2]
table_sub

# <단계2> 생성된 데이터프레임 대상 칼럼명 수정 : ['seoul','incheon','busan']
table_sub.columns =['seoul','incheon','busan']
table_sub

# <단계3> 생성된 데이터프레임 대상  index 수정 : ['male', 'female']     
table_sub.index =['male', 'female']  
table_sub

# <단계4> 생성된 데이터프레임 대상 누적가로막대차트 그리기
table_sub.plot(kind='barh', stacked = True)
plt.title('gender vs size')


