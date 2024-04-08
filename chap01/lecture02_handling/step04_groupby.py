# -*- coding: utf-8 -*-
"""
step04_groupby
"""

import pandas as pd 

 
path = r'C:\ITWILL\4_Python_ML\data'

# dataset load & 변수 확인
wine = pd.read_csv(path  + '/winequality-both.csv')
print(wine.info())
'''
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
0   type                  6497 non-null   object : 집단변수    
 :
12  quality               6497 non-null   int64 : 와인품질 
'''    

# 칼럼 공백 -> '_' 교체 
wine.columns = wine.columns.str.replace(' ', '_')
wine.head()
print(wine.info())


# 5개 변수 선택 : subset 만들기 
wine_df = wine.iloc[:, [0,1,4,11,12]] # 위치 기반
print(wine_df.info()) 

# 특정 칼럼명 수정 
columns = {'fixed_acidity':'acidity', 'residual_sugar':'sugar'} # {'old','new'} 
wine_df = wine_df.rename(columns = columns) 
wine_df.info()
    
# 집단변수 확인 : 와인유형   
print(wine_df.type.unique()) # ['red' 'white']
print(wine_df.type.nunique()) # 2

# 이산변수 확인 : 와인 품질    
print(wine_df.quality.unique()) # [5 6 7 4 8 3 9]
print(wine_df.quality.value_counts())


# 1. 범주형 변수 기준 subset 만들기 

# 1) 1개 집단 기준  
red_wine = wine_df[wine['type']=='red']  

# 2) 2개 이상 집단 기준 
two_wine_type = wine_df[wine_df['type'].isin(['red','white'])] 

# 3) 범주형 변수 기준 특정 칼럼 선택 : 1차원 구조
red_wine_quality = wine.loc[wine['type']=='red', 'quality']  
white_wine_quality = wine.loc[wine['type']=='white', 'quality'] 


# 2. 범주형 변수 기준 group & 통계량

# 1) 범주형변수 1개 이용 그룹화 
type_group = wine_df.groupby('type')

# 각 집단별 빈도수 
type_group.size()  

# 그룹객체에서 그룹 추출 
red_df = type_group.get_group('red')
white_df = type_group.get_group('white')

    
# 그룹별 통계량 
print(type_group.sum()) 
print(type_group.mean())


# 2) 집단변수 2개 이용 : 나머지 변수(3개)가 그룹 대상 
wine_group = wine_df.groupby(['type','quality']) # 2개 x 7개 = 최대 14  

# 각 집단별 빈도수
wine_group.size()

# 그룹 통계 시각화 
grp_mean = wine_group.mean()


# 3. apply() 함수 

# 1) 사용자 함수 : 0 ~ 1 사이 정규화 
def normal_df(x):
    nor = ( x - min(x) ) / ( max(x) - min(x) )
    return nor


# 2) 2차원 data 준비 : wine 데이터 적용 
wine_x = wine_df.iloc[:, 1:] # 숫자변수만 선택 


# 3) apply 함수 적용 : 열(칼럼) 단위로 실인수 전달   
wine_nor = wine_x.apply(normal_df) 
print(wine_nor.describe()) # 정규화 확인 


# 4. map() 함수   

# 1) 인코딩 함수 
def encoding_df(x):
    encoding = {'red':[1,0], 'white':[0,1]}
    return encoding[x]

# 2) 1차원 data 준비 
wine_type = wine_df['type']


# 3) map 함수 적용 
label = wine_type.map(encoding_df)








