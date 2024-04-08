# -*- coding: utf-8 -*-
"""
lecture02 > step04 관련문제

문7) titanic 데이터셋을 대상으로 아래와 같이 단계별로 처리하시오. 
"""

import seaborn as sns # seaborn 데이터셋 이용 

titanic = sns.load_dataset('titanic')
titanic.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
'''
 

# 단계1 : age, sex, class, fare, survived 칼럼으로 서브셋 생성 


# 단계2 : class와 sex 칼럼 기준으로 그룹 객체 생성 및 크기 확인 


# 단계3 : 그룹별 평균 구하기 


# 단계4 : 그룹별 평균에서 survived 칼럼 기준 막대차트 시각화  

