'''
lecture01 > step02 관련문제

문3) wdbc_data.csv 파일을 읽어와서 단계별로 x, y 변수를 생성하시오.
     <단계1> 파일 가져오기, 정보 확인 
     <단계2> x,y변수 선택 : 
            y변수 : diagnosis 
            x변수 : id 칼럼 제외 나머지 30개 칼럼
     <단계3> y변수의 범주('B', 'M')를 기준으로 서브셋 만들기
            B_tumor = 'B'범주를 갖는 서브셋 
            M_tumor = 'M'범주를 갖는 서브셋             
'''
import pandas as pd

path = r"c:/ITWILL/4_Python_ML/data" # file 경로 변경 

# <단계1> 파일 가져오기, 정보 확인 


# <단계2> y변수, x변수 선택
y = None # diagnosis 칼럼 
x = None # id 칼럼 제외 나머지 30개 칼럼

# <단계3> y변수의 범주('B', 'M')를 기준으로 서브셋 만들기
B_tumor = None
M_tumor = None 