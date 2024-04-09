# -*- coding: utf-8 -*-
"""
step01_DF_merge.py

merge() : 공통칼럼 있어야 함 
concat() : 공통칼럼 없어도 됨
<parameter>
 * merge() 
on = ['공통칼럼1','공통칼럼2']
how = 'inner' or 'outer'
 * concat() 
objs =[데이터프레임1,데이터프레임2]
-------------------
사용 데이터 : wdbc_data.csv 
Wisconsin Diagnostic Breast Cancer (위스콘신)

범주형
M : Malignant (악성) 
B : Benign (양성)
"""

import pandas as pd 
pd.set_option('display.max_columns', 100) # 콘솔에서 보여질 최대 칼럼 개수 

path = r'C:\ITWILL\4_Python_ML\data'

wdbc = pd.read_csv(path + '/wdbc_data.csv')
wdbc.info()
'''
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):
'''


# 전체 칼럼 가져오기 (32개)
# 밑에 데이터프레임에서 여러 칼럼 선택하려고 ★★★
cols = list(wdbc.columns)
print(cols)

# 1. DF 병합(merge) # sql join 과 비슷한 과정
DF1 = wdbc[cols[:16]] # id 칼럼 존재 
DF2 = wdbc[cols[16:]] # id 칼럼 x 

# 공통칼럼이 있어야 하므로, id 추가 ★★★
DF2['id'] = wdbc.id # DF2는 칼럼이 17개가 됨 ★★

DF3 = pd.merge(left=DF1, right=DF2, on='id') 
DF3.shape # >> (569, 32)


# 2. DF 결합(concat) - 공통칼럼이 없는 경우★
DF2 = wdbc[cols[16:]]

DF4 = pd.concat(objs=[DF1, DF2], axis = 1) 
# axis = 1 : 열축 기준 결합 cbind
DF4.shape # >> (569, 32)

# 3. Inner join과 Outer join 
name = ['hong','lee','park','kim']
age = [35, 20, 33, 50]

df1 = pd.DataFrame(data = {'name':name, 'age':age}, 
                   columns = ['name', 'age'])

name2 = ['hong','lee','kim']
age2 = [35, 20, 50]
pay = [250, 350, 250]

df2 = pd.DataFrame(data = {'name':name2, 'age':age2,'pay':pay}, 
                   columns = ['name', 'age', 'pay'])

inner = pd.merge(left=df1, right=df2, how='inner')
inner.shape # >> (3,3)
outer = pd.merge(left=df1, right=df2, how='outer')  
outer.shape # >> (4,3)

outer
'''   name  age    pay
0  hong   35  250.0
1   lee   20  350.0
2  park   33    NaN
3   kim   50  250.0
'''