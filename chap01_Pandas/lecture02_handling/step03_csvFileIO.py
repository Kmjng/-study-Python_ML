# -*- coding: utf-8 -*-
"""
step03_csvFileIO.py
"""

import pandas as pd 

path = r'C:\ITWILL\4_Python_ML\data'

# 1. csv file read

# 1) 칼럼명이 없는 경우 
st = pd.read_csv(path + '/student.csv', header=None)
st # 0     1    2   3 -> 기본 칼럼명 

# 칼럼명 수정 
col_names = ['sno','name','height','weight'] # list 
st.columns = col_names # 칼럼 수정 
print(st)
'''
   sno  name  height  weight        bmi
0  101  hong     175      65  21.224490
1  201   lee     185      85  24.835646
2  301   kim     173      60  20.047446
3  401  park     180      70  21.604938
'''

# 2) 칼럼명 특수문자(.) or 공백 
iris = pd.read_csv(path + '/iris.csv')
print(iris.info())

#iris.Sepal.Length # AttributeError

# 점(.) -> 언더바(_) 교체 
# 데이터프레임 이름 변경하려면, 문자열로 접근해야 한다. 
# replace() (X)
# str.replace() 메서드
# 문자열에 포함된 특정 패턴을 다른 패턴으로 대체
type(iris.columns) # pandas.core.indexes.base.Index
type(iris.columns.str) # pandas.core.strings.accessor.StringMethods
iris.columns = iris.columns.str.replace('.','_') # ('old','new')

iris.info() # Sepal_Length
iris.Sepal_Length


# 3) 특수구분자(tab키), 천단위 콤마 
# pd.read_csv('file', delimiter='\t', thousands=',')


# 2. data 처리 : 파생변수 추가 
'''
비만도 지수(bmi) = 몸무게/(키**2)
'''

bmi = st.weight / (st.height*0.01)**2
bmi
    
# 파생변수 추가 
st['bmi'] = bmi


''' 파생변수에 대한 범주형 변수 생성하기 (명목척도) ★★★
label : normal, fat, thin 
normal : bmi : 18 ~ 23
fat : 23 초과
thin : 18 미만  
'''
label =[] 
for bmi in st.bmi : # st데이터프레임의 bmi 칼럼
    if bmi >= 18 and bmi <=23 : 
        label.append('normal')
    elif bmi > 23: 
        label.append('fat')
    else: 
        label.append('thin')
st['label']=label
st
'''   sno  name  height  weight        bmi   label
0  101  hong     175      65  21.224490  normal
1  201   lee     185      85  24.835646     fat
2  301   kim     173      60  20.047446  normal
3  401  park     180      70  21.604938  normal
'''

# 3. csv file 저장 # st_info.csv
dir(st)
'''
to_csv 
to_excel
to_json
'''
st.to_csv(path + '/st_info.csv', index = None, encoding='utf-8')

