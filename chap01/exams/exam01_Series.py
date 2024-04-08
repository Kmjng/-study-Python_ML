''' 
lecture01 > step01 관련문제

문1) product1과 product2의 Series 객체를 이용하여 다음과 같은 단계로 다음과 처하시오.
   <단계1> Series 연산 : product3 = product1 - product2 
   <단계2> product3 결측치 유무 확인 
   <단계3> product3 결측치 채우기 : product3 최댓값 - product3 최솟값
   <단계4> product4 만들기 : product3의 평균이하 자료만 추출하여 product4 생성 
    
  
<<각 단계별 출력 결과>>

<단계1> Series 연산
apple     3000.0
banana       NaN
kiwi      3000.0
mango     1000.0
orange       NaN
dtype: float64

<단계2> product3 결측치 유무 확인
apple     False
banana     True
kiwi      False
mango     False
orange     True
dtype: bool

<단계3> product3 결측치 채우기
apple     3000.0
banana    2000.0
kiwi      3000.0
mango     1000.0
orange    2000.0
dtype: float64

<단계4> product4 만들기
banana    2000.0
mango     1000.0
orange    2000.0
'''

from pandas import Series
import pandas as pd 

product1 = Series([6000, 3500, 1500, 5000], 
                  index=['apple', 'mango','orange', 'kiwi'])
product2 = Series([3000, 3000, 2500, 2000], 
                  index=['apple', 'banana','mango', 'kiwi'])


# <단계1> Series 연산 : product3 = product1 - product2 

# <단계2> product3 결측치(NaN) 유무 확인 

# <단계3> product3 결측치(NaN) 채우기 : 결측치 = product3 최댓값 - product3 최솟값

# <단계4> product4 만들기 : product3의 평균 이하의 자료만 추출 






