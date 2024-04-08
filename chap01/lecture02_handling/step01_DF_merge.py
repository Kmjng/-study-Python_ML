# -*- coding: utf-8 -*-
"""
step01_DF_merge.py
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

# 전체 칼럼 가져오기 
cols = list(wdbc.columns)


# 1. DF 병합(merge)
DF1 = wdbc[cols[:16]] 
DF2 = wdbc[cols[16:]] 

DF3 = pd.merge(left=DF1, right=DF2, on='id') 


# 2. DF 결합(concat)
DF2 = wdbc[cols[16:]]

DF4 = pd.concat(objs=[DF1, DF2], axis = 1) # 열축 기준 결합


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

outer = pd.merge(left=df1, right=df2, how='outer')  













