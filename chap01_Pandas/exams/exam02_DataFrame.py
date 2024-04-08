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

path = r"C:/ITWILL/4_Python_ML/data" # file 경로 변경 

score = pd.read_csv(path + '/score.csv')
print(score.info())
print(score)
'''
  name  score   iq  academy  game  tv
0    A     90  140        2     1   0
1    B     75  125        1     3   3
2    C     77  120        1     0   4
3    D     83  135        2     3   2
4    E     65  105        0     4   4
5    F     80  123        3     1   1
6    G     83  132        3     4   1
7    H     70  115        1     1   3
8    I     87  128        4     0   0
9    J     79  131        2     2   3
'''

# <단계1> tv 칼럼이 0인 관측치 2개 삭제
new_df = score[score.tv != 0] 

new_df.shape # 차원 확인 
# (8,6)


# <단계2> score, academy 칼럼만 추출하여 DataFrame 생성
new_df2 = new_df[['score','academy']]

new_df2.shape # 차원 확인 
# (8,2)

# <단계3> new_df2 이용 score, academy의 평균 계산  
new_df2.mean(axis=0)
'''
score      76.500
academy     1.625
dtype: float64
'''

