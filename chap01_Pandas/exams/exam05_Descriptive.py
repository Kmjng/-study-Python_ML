'''  
lecture01 > step03 관련문제

문5) tips.csv 파일을 읽어와서 다음과 같이 처리하시오.
   <단계1> 파일 정보 보기 
   <단계2> header를 포함한 앞부분 5개 관측치 보기 
   <단계3> header를 포함한 뒷부분 5개 관측치 보기 
   <단계4> 숫자 칼럼 대상 요약통계량 보기 
   <단계5> 흡연자(smoker) 유무 빈도수 계산  
   <단계6> 요일(day) 칼럼의 유일한 값 출력 
'''

import pandas as pd

path = r"c:\ITWILL\4_Python_ML\data" # file 경로 변경

tips = pd.read_csv(path + '/tips.csv')

# <단계1> 파일 정보 보기
tips.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    object 
 3   smoker      244 non-null    object 
 4   day         244 non-null    object 
 5   time        244 non-null    object 
 6   size        244 non-null    int64  
dtypes: float64(2), int64(1), object(4)
memory usage: 13.5+ KB
'''
# <단계2> header를 포함한 앞부분 5개 관측치 보기 
tips.head(5)
'''   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
'''
# <단계3> header를 포함한 뒷부분 5개 관측치 보기 
tips.tail(5)
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    object 
 3   smoker      244 non-null    object 
 4   day         244 non-null    object 
 5   time        244 non-null    object 
 6   size        244 non-null    int64  
dtypes: float64(2), int64(1), object(4)
memory usage: 13.5+ KB
Out[15]: '   total_bill   tip     sex smoker  day    time  size\n0       16.99  1.01  Female     No  Sun  Dinner     2\n1       10.34  1.66    Male     No  Sun  Dinner     3\n2       21.01  3.50    Male     No  Sun  Dinner     3\n3       23.68  3.31    Male     No  Sun  Dinner     2\n4       24.59  3.61  Female     No  Sun  Dinner     4'

tips.tail(5)
Out[16]: 
     total_bill   tip     sex smoker   day    time  size
239       29.03  5.92    Male     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
241       22.67  2.00    Male    Yes   Sat  Dinner     2
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2
'''
# <단계4> 숫자 칼럼 대상 요약통계량 보기 
tips[['total_bill', 'tip', 'size']].info()
'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   size        244 non-null    int64  
dtypes: float64(2), int64(1)
memory usage: 5.8 KB
'''
# <단계5> 흡연자(smoker) 유무 빈도수 계산  
tips.value_counts(tips.smoker=='Yes')
'''smoker
False    151
True      93
Name: count, dtype: int64
'''
# <단계6> 요일(day) 칼럼의 유일한 값 출력 
tips.day.unique()
'''array(['Sun', 'Sat', 'Thur', 'Fri'], dtype=object)'''
