# -*- coding: utf-8 -*-
"""
Seaborn의 두가지 문자열 타입
1. Object vs Category 
  - object : 문자열. 여러 타입이 object가 될 수 있음. 
              순서 변경 불가 
  - category : 문자열. 범주형으로 순서형(ordinal)범주로 주로 사용됨.  
              순서 변경 가능 
2. 범주형 자료 시각화 
"""

import seaborn as sn


# 1. Object vs Category 

# dataset load
titanic = sn.load_dataset('titanic')

print(titanic.info())

# subset 만들기 
df = titanic[['survived','age','class','who']]
df.info()
'''
 0   survived  891 non-null    int64   
 1   age       714 non-null    float64 
 2   class     891 non-null    category
 3   who       891 non-null    object 
'''
df.head()
'''
   survived   age  class    who
0         0  22.0  Third    man
1         1  38.0  First  woman
2         1  26.0  Third  woman
3         1  35.0  First  woman
4         0  35.0  Third    man
'''

# category형 정렬 
df.sort_values(by = 'class') # category 오름차순
# First > Second > Third

# object형 정렬 
df.sort_values(by = 'who') # object 오름차순 
# child > man > woman


# category형 변수 특정 순서 변경 
# cat.set_categories(['내용1','내용2',..])
# 범주형 데이터에 새로운 범주를 설정 
df['class_new'] = df['class'].cat.set_categories(['Third', 'First', 'Second'])
# reorder_categories()  
# 범주형 데이터의 순서를 변경
df['class_new'] = df['class'].cat.reorder_categories(['Third', 'First', 'Second'])

df
'''
     survived   age   class    who class_new who_new
0           0  22.0   Third    man     Third     man
1           1  38.0   First  woman     First   woman
2           1  26.0   Third  woman     Third   woman
3           1  35.0   First  woman     First   woman
4           0  35.0   Third    man     Third     man
..        ...   ...     ...    ...       ...     ...
886         0  27.0  Second    man    Second     man
887         1  19.0   First  woman     First   woman
888         0   NaN   Third  woman     Third   woman
889         1  26.0   First    man     First     man
890         0  32.0   Third    man     Third     man

[891 rows x 6 columns]
'''

# object -> category 형 변환 
df['who'].dtype # >> object
df['who_new'] = df['who'].astype('category')

df['who_new'] = df['who_new'].cat.set_categories(['man','woman','child'])
# reorder_categories() 
df['who_new'] = df['who_new'].cat.reorder_categories(['man','woman','child'])

df

'''
     survived   age   class    who class_new who_new
0           0  22.0   Third    man     Third     man
1           1  38.0   First  woman     First   woman
2           1  26.0   Third  woman     Third   woman
3           1  35.0   First  woman     First   woman
4           0  35.0   Third    man     Third     man
..        ...   ...     ...    ...       ...     ...
886         0  27.0  Second    man    Second     man
887         1  19.0   First  woman     First   woman
888         0   NaN   Third  woman     Third   woman
889         1  26.0   First    man     First     man
890         0  32.0   Third    man     Third     man

[891 rows x 6 columns]
'''
# 2. 범주형 자료 시각화 

# 1) 배경 스타일 
sn.set_style(style='darkgrid')
tips = sn.load_dataset('tips')
print(tips.info())

# 2) category형 자료 시각화 
# countplot(x= , data= )
import matplotlib.pyplot as plt
sn.countplot(x = 'smoker', data = tips) 
plt.title('smoker of tips')
plt.show()
