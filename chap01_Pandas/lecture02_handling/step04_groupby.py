# -*- coding: utf-8 -*-
"""
step04_groupby

input data >> split by key >> apply(sum) >> combine

1. 범주형 변수 기준 subset 만들기 
2. 범주형 변수 기준 groupby & 통계량
3. apply() 함수 : DataFrame(2D) 객체에 함수 적용 
4. map() 함수 : Series(1D) 객체에 함수 적용

"""

import pandas as pd 

 
path = r'C:\ITWILL\4_Python_ML\data'

# dataset load & 변수 확인
wine = pd.read_csv(path  + '/winequality-both.csv')
print(wine.info())
'''
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
0   type                  6497 non-null   object => 와인유형 (범주형)    
 :
12  quality               6497 non-null   int64 : 와인품질 (이산형) 
'''

# 칼럼 공백 -> '_' 교체 # str.replace()
wine.columns = wine.columns.str.replace(' ', '_')
wine.head()
print(wine.info())


# 5개 변수 선택 : subset 만들기 
wine_df = wine.iloc[:, [0,1,4,11,12]] # 13개 중 5개 열 선택
print(wine_df.info()) 

# 특정 칼럼명 수정 ★★★★
'''
방법 1.
DF.columns =['칼렴명',...] : 전체칼럼 수정 
방법 2. 
DF.rename(columns = {'기존':'변경'}) # dict형으로 수정가능
'''
columns = {'fixed_acidity':'acidity', 'residual_sugar':'sugar'} # {'old','new'} 
wine_df = wine_df.rename(columns = columns) 
wine_df.info()
    
# 집단변수 확인 : 와인유형   
# unique() 
# nunique() unique한 값 갯수 ★★★
print(wine_df.type.unique()) # ['red' 'white']
print(wine_df.type.nunique()) # 2
wine_df.type.value_counts()
'''type
white    4898
red      1599
'''


# 이산변수 확인 : 와인 품질    
print(wine_df.quality.unique()) # [5 6 7 4 8 3 9]
print(wine_df.quality.value_counts())


# 1. 범주형 변수 기준 subset 만들기 

# 1) 1개 집단 기준  
red_wine = wine_df[wine['type']=='red']  
red_wine.shape # >> (1599,5)

white_wine = wine_df[wine['type']=='white']  
white_wine.shape # >> (4898, 5)


# 2) 2개 이상 집단 기준 
# type 칼럼이 red or white 인 값만 가져오기
two_wine_type = wine_df[wine_df['type'].isin(['red','white'])] 

# 3) 범주형 변수 기준 특정 칼럼 선택 : 1차원 구조
# loc[] 명칭기반에 조건식 가능 ★★★
# loc[조건식, 조회할칼럼] : 조건식에 해당하는 값만 출력해줌
red_wine_quality = wine.loc[wine['type']=='red', 'quality']  
white_wine_quality = wine.loc[wine['type']=='white', 'quality'] 


# 2. 범주형 변수 기준 groupby => 통계량 데이터만 확인 가능하다. 
'''<pandas.core.groupby.generic.DataFrameGroupBy 
object at 0x000001DE155A1710>
'''
# 1) 범주형변수('type') 1개 이용 그룹화
type_group = wine_df.groupby('type')

# 각 집단별 빈도수 
# unique()대신 size() ★★
type_group.size()  
'''
type
red      1599
white    4898
dtype: int64
'''
# 그룹객체에서 그룹 추출 
# get_group('값1')
red_df = type_group.get_group('red')
white_df = type_group.get_group('white')

    
# 그룹별 통계량 
print(type_group.sum()) 
'''
        acidity     sugar    alcohol   quality
type                                          
red    8.319637  2.538806  10.422983  5.636023
white  6.854788  6.391415  10.514267  5.877909
'''
print(type_group.mean())


# 2) 범주형변수 2개 ('type','quality') 이용 : 나머지 변수(3개)가 그룹 대상 
wine_group = wine_df.groupby(['type','quality']) # 2개 x 7개 = 최대 14  

# 각 집단별 빈도수
wine_group.size()
'''
type   quality
red    3            10
       4            53
       5           681
       6           638
       7           199
       8            18
white  3            20
       4           163
       5          1457
       6          2198
       7           880
       8           175
       9             5
dtype: int64'''
# 그룹 통계 시각화 
grp_mean = wine_group.mean()
grp_mean
'''
                acidity     sugar    alcohol
type  quality                               
red   3        8.360000  2.635000   9.955000
      4        7.779245  2.694340  10.265094
      5        8.167254  2.528855   9.899706
      6        8.347179  2.477194  10.629519
      7        8.872362  2.720603  11.465913
      8        8.566667  2.577778  12.094444
white 3        7.600000  6.392500  10.345000
      4        7.129448  4.628221  10.152454
      5        6.933974  7.334969   9.808840
      6        6.837671  6.441606  10.575372
      7        6.734716  5.186477  11.367936
      8        6.657143  5.671429  11.636000
      9        7.420000  4.120000  12.180000
'''
grp_mean.plot(kind = 'bar')

# 3. apply() 함수 

# 1) 사용자 함수 : 0 ~ 1 사이 정규화 
# MinMax Normalization
def normal_df(x):
    nor = ( x - min(x) ) / ( max(x) - min(x) )
    return nor

wine_df
# 2) 2차원 data 준비 : wine 데이터 적용 
wine_x = wine_df.iloc[:, 1:] # 숫자변수만 선택 
wine_x.shape # (6497,4)
wine_x

# 3) apply 함수 적용 : 열(칼럼) 단위로 실인수 전달해서 적용됨  
# DF명.apply(함수명 또는 사용자함수명) 
wine_nor = wine_x.apply(normal_df) 
print(wine_nor.describe()) # 요약통계량으로 정규화된 값 확인하기 
''' ** 정규화한 값들임
           acidity        sugar      alcohol      quality
count  6497.000000  6497.000000  6497.000000  6497.000000
mean      0.282257     0.074283     0.361131     0.469730
std       0.107143     0.072972     0.172857     0.145543
min       0.000000     0.000000     0.000000     0.000000
25%       0.214876     0.018405     0.217391     0.333333
50%       0.264463     0.036810     0.333333     0.500000
75%       0.322314     0.115031     0.478261     0.500000
max       1.000000     1.000000     1.000000     1.000000
'''

# 4. map() 함수   
# 시리즈명.map(함수명)
# 1) 인코딩 함수 
def encoding_df(x):
    encoding = {'red':[1,0], 'white':[0,1]}
    return encoding[x] # x에 'red','white'가 들어감

# 2) 1차원 data 준비 
wine_type = wine_df['type']


# 3) map 함수 적용 
label = wine_type.map(encoding_df)
label
'''
0       [1, 0]
1       [1, 0]
2       [1, 0]
...
6495    [0, 1]
6496    [0, 1]
Name: type, Length: 6497, dtype: object
'''
# 인라인방식 (lambda)
encoding = {'red':[1,0],'white':[0,1]} # dict # mapping table 
label2 = wine_type.map(lambda x: encoding[x] )
label2

label3 = wine_type.map(lambda x: {'red':[1,0],'white':[0,1]}[x])
label3

wine_df['label']=label
wine_df
'''
       type  acidity  sugar  alcohol  quality   label
0       red      7.4    1.9      9.4        5  [1, 0]
1       red      7.8    2.6      9.8        5  [1, 0]
...
6495  white      5.5    1.1     12.8        7  [0, 1]
6496  white      6.0    0.8     11.8        6  [0, 1]

[6497 rows x 6 columns]
'''