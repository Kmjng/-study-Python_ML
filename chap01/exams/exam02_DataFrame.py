''' 
lecture01 > step02 관련문제

문2) score.csv 파일을 읽어와서 다음과 같은 단계로 처리하시오.
   <단계1> tv 칼럼이 0인 관측치 2개 삭제 (조건식 이용)
   <단계2> score, academy 칼럼만 추출하여 DataFrame 생성
   <단계3> score, academy 칼럼의 평균 계산 : <<출력 결과 >> 참고 
       
   
<<출력 결과 >>
score      76.500
academy     1.625   
'''

import pandas as pd

path = r"c:\ITWILL\4_Python_ML\data" # file 경로 변경 

score = pd.read_csv(path + '/score.csv')
print(score.info())
print(score)


# <단계1> tv 칼럼이 0인 관측치 2개 삭제
new_df = None 

new_df.shape # 차원 확인 


# <단계2> score, academy 칼럼만 추출하여 DataFrame 생성
new_df2 = None

new_df2.shape # 차원 확인 


# <단계3> new_df2 이용 score, academy의 평균 계산  













