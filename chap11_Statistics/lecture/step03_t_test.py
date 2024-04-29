'''
t검정 : t 분포에 대한 가설검정  
  1. 단일표본 t검정 : 한 집단 평균차이 검정  
  2. 독립표본 t검정 : 두 집단 평균차이 검정
  3. 대응표본 t검정 : 대응 두 집단차이 검정 
'''

from scipy import stats # test
import numpy as np # sampling
import pandas as pd # csv file read


### 1. 단일표본 t검정 : 한 집단 평균차이 검정   

# 대립가설(H1) : 모평균(mu) ≠ 174
# 귀무가설(H0) : 모평균(mu) = 174 

# 남자 평균 키  170cm ~ 180cm -> 29명 표본추출 
sample_data = np.random.uniform(170,180, size=29) 
print(sample_data)

# 기술통계 
print('평균 키 =', sample_data.mean()) 

# 단일집단 평균차이 검정 
one_group_test = stats.ttest_1samp(sample_data, 174, alternative='two-sided') 
print('t검정 통계량 = %.3f, pvalue = %.5f'%(one_group_test))



### 2. 독립표본 t검정 :  두 집단 평균차이 검정

# 대립가설(H1) : 남자평균점수 < 여자평균점수
# 귀무가설(H0) : 남여 평균 점수에 차이가 없다.

np.random.seed(36)
male_score = np.random.uniform(45, 95, size=30) # 남성 
female_score = np.random.uniform(50, 100, size=30) # 여성 


two_sample = stats.ttest_ind(male_score, female_score)
print(two_sample)
print('두 집단 평균 차이 검정 = %.3f, pvalue = %.3f'%(two_sample))


# file 자료 이용 : 교육방법에 따른 실기점수의 평균차이 검정  

# 대립가설(H1) : 교육방법에 따른 실기점수의 평균에 차이가 있다.
# 귀무가설(H0) : 교육방법에 따른 실기점수의 평균에 차이가 없다.

sample = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\two_sample.csv')
print(sample.info())

two_df = sample[['method', 'score']]
print(two_df)


# 교육방법 기준 subset
method1 = two_df[two_df.method==1]
method2 = two_df[two_df.method==2]


# score 칼럼 추출 
score1 = method1.score
score2 = method2.score


# 두 집단 평균차이 검정 
two_sample = stats.ttest_ind(score1, score2)
print(two_sample)


### 3. 대응표본 t검정 : 대응 두 집단 평균차이 검정

# 대립가설(H1) : 복용전과 복용후 몸무게 차이가 0 보다 크다.(복용전 몸무게 > 복용후 몸무게)
# 귀무가설(H0) : 복용전과 복용후 몸무게 차이에 변화가 없다.

before = np.random.randint(60, 65, size=30)  
after = np.random.randint(59, 64,  size=30) 

paired_sample = stats.ttest_rel(before, after)
print(paired_sample)
print('t검정 통계량 = %.5f, pvalue = %.5f'%paired_sample)



