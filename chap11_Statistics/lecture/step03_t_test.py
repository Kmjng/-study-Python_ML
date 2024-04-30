'''
t검정 : t 분포에 대한 가설검정  
  1. 단일표본 t검정 : 한 집단 평균차이 검정  
  2. 독립표본 t검정 : 두 집단 평균차이 검정
  3. 대응표본 t검정 : 대응 두 집단차이 검정 

★★ t통계량을 구해서 비교하는 원리 

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
# 평균 키 = 175.00518013116923

# 단일집단 평균차이 검정 
# stats.ttest_1samp()

one_group_test = stats.ttest_1samp(sample_data, 174, 
                                   alternative='two-sided') 
print('t검정 통계량 = %.3f, pvalue = %.5f'%(one_group_test))
# t검정 통계량 = 1.845, pvalue = 0.07571
# 자유도(n-1) = 29 -1
one_group_test
'''
TtestResult(statistic=1.8445208855399147,
            pvalue=0.0757114740072547, 
            df=28)
'''
alpha = 0.05 
pvalue =one_group_test[1] # 0.0757114740072547
# 또는 pvalue = one_group_test.pvalue ★★★

if pvalue > alpha : 
    print('귀무가설 채택; 모평균은 174 이다. 174와 차이가 없다.')
else : 
    print("귀무가설 기각")



### 2. 독립표본 t검정 :  '두 집단' 평균차이 검정

# 대립가설(H1) : 남자평균점수 < 여자평균점수 (실습을 위한 가설설정임)
# 귀무가설(H0) : 남여 평균 점수에 차이가 없다.

np.random.seed(36)
male_score = np.random.uniform(45, 95, size=30) # 남성 
female_score = np.random.uniform(50, 100, size=30) # 여성 


two_sample = stats.ttest_ind(male_score, female_score, 
                             alternative ='less')
'''
stats.ttest_ind(a,b, alternative ='less') # a < b 
stats.ttest_ind(a,b, alternative ='greater') # a > b 
'''
print(two_sample)
print('두 집단 평균 차이 검정 = %.3f, pvalue = %.3f'%(two_sample))
'''
두 집단 평균 차이 검정 = -2.139, pvalue = 0.018
'''



# file 자료 이용 : 교육방법에 따른 실기점수의 평균차이 검정  

# 대립가설(H1) : 교육방법에 따른 실기점수의 평균에 차이가 있다.
# 귀무가설(H0) : 교육방법에 따른 실기점수의 평균에 차이가 없다.

sample = pd.read_csv(r'C:/ITWILL/4_Python_ML/data/two_sample.csv')
print(sample.info())

two_df = sample[['method', 'score']]
print(two_df)
'''
     method  score
0         1    5.1
1         1    5.2
2         1    4.7
3         1    NaN
..      ...    ...
235       2    NaN
236       2    5.4
237       2    6.0
238       2    6.7
239       2    5.2

[240 rows x 2 columns]
'''

# 결측치 : 평균으로 대체 
two_df.isnull().sum()

two_df.score = two_df.score.fillna(two_df.score.mean())


# 교육방법 기준 subset
method1 = two_df[two_df.method==1] # 교육방법 1
method2 = two_df[two_df.method==2] # 교육방법 2


# score 칼럼 추출 
score1 = method1.score #시리즈 형태의 표본
score2 = method2.score 


# 두 집단 평균차이 검정 
two_sample = stats.ttest_ind(score1, score2) 
print(two_sample)


### 3. '대응표본' t검정 : 대응 두 집단 평균차이 검정
# stats.ttest_rel() ; relative

# 다이어트 식품이라면 ? 
# 복용전 - 복용 후 > 0 가 되어야 함 (대립가설)
# 대립가설(H1) : 복용전과 복용후 몸무게 차이가 0 보다 크다.(복용전 몸무게 > 복용후 몸무게)
# 귀무가설(H0) : 복용전과 복용후 몸무게 차이에 변화가 없다. (없거나 복용후가 더 큼)

before = np.random.randint(60, 65, size=30)  
after = np.random.randint(59, 64,  size=30) 

difference = before.mean() - after.mean()
difference # 0.43333333333333

paired_sample = stats.ttest_rel(before, after, alternative='greater')
print(paired_sample)
print('t검정 통계량 = %.5f, pvalue = %.5f'%paired_sample)
'''
<alternative 양측검정으로 하면>
t검정 통계량 = 2.67953, pvalue = 0.01202
>> 대립가설 채택 : 차이가 있다. 

<alternative 단측검정으로 하면> 
t검정 통계량 = 2.86488, pvalue = 0.00384
>> 대립가설 채택 : 전 > 후 다. 

'''


