'''
 카이제곱 검정(chisquare test) 
 * 적합도 검정 (1개의 변인의 적합성 검정) 
 * 독립성 검정 (2개의 변인 간 독립성 검정) 
 
  - 검정통계량(기대비율) = sum( (관측값 - 기댓값)**2 / 기댓값 )
  - 관측값과 기댓값 차이가 없으면 귀무가설 채택 
  
  
  자유도(df) = (행의 수 - 1) * (열의 수 - 1)
'''

from scipy import stats # 확률분포 검정 


### 1. 일원 chi-square(1개 변수 이용) : 적합성 검정 
# ★★★★
# 일원카이제곱검정 주의사항 : 관측값 합과 기댓값 합이 동일해야 한다. 
'''
 귀무가설 : 관측치와 기대치는 차이가 없다. ★★★
 >> 주어진 데이터가 특정 이론적 분포와 적합하다는 것
 대립가설 : 관측치와 기대치는 차이가 있다. 
'''

# 주사위 적합성 검정 
real_data = [4, 6, 17, 16, 8, 9] # 관측값; (관측도수) 
exp_data = [10,10,10,10,10,10] # 기대값; (기대도수)
chis = stats.chisquare(real_data, exp_data)
print(chis)
print('statistic = %.3f, pvalue = %.3f'%(chis)) 
# statistic = 14.200, pvalue = 0.014
'''
유의수준 0.05에 의해 귀무가설 기각 
적합하지 않다. 

'''
sum(real_data) # 60 
sum(exp_data) # 60 
# 만약 합이 같지 않으면 비율화 해서 계산할 것 




### 2. 이원 chi-square(2개 변수 이용) : 교차행렬의 관측값과 기대값으로 검정
'''
 귀무가설 : 교육수준과 흡연율 간에 관련성이 없다. ★★★(두 변인은 독립이다)
 대립가설 : 교육수준과 흡연율 간에 관련성이 있다.
'''

# 파일 가져오기
import pandas as pd

path = r'C:\ITWILL\4_Python_ML\data'
smoke = pd.read_csv(path + "/smoke.csv")
smoke.info()

# <단계 1> 변수 선택 
print(smoke)# education, smoking 변수
education = smoke.education 
smoking = smoke.smoking
 
smoking.unique() # 3
education.unique() # 3


# <단계 2> 교차분할표 
tab = pd.crosstab(index=education, columns=smoking)
print(tab) # 관측값 


# <단계3> 카이제곱 검정 : 교차분할표 이용 
chi2, pvalue, df, evalue = stats.chi2_contingency(observed= tab)  

# chi2 검정통계량, 유의확률, 자유도, 기대값  
print('chi2 = %.6f, pvalue = %.6f, d.f = %d'%(chi2, pvalue, df))
'''
chi2 = 18.910916, pvalue = 0.000818, d.f = 4
'''

# <단계4> 기대값 
print(evalue)
'''
smoking     1   2   3
education            
1          51  92  68
2          22  21   9
3          43  28  21
chi2 = 18.910916, pvalue = 0.000818, d.f = 4
[[68.94647887 83.8056338  58.24788732]
 [16.9915493  20.65352113 14.35492958]
 [30.06197183 36.54084507 25.3971831 ]]
'''

#############################################
# 성별과 흡연 간의 독립성 검정 example 
#############################################
'''
 귀무가설 : 성별과 흡연유무 간에 관련성이 없다.
 대립가설 : 성별과 흡연유무 간에 관련성이 있다.
'''
import seaborn as sn
import pandas as pd

# <단계1> titanic dataset load 
tips = sn.load_dataset('tips')
print(tips.info())
sex = tips.sex
smoker = tips.smoker
# <단계2> 교차분할표 
tab = pd.crosstab(index = sex, columns = smoker)
tab
'''
smoker  Yes  No
sex            
Male     60  97
Female   33  54
'''
# <단계3> 카이제곱 검정 
chi2, pvalue, df, evalue = stats.chi2_contingency(observed= tab)
chi2 # 0.0000
pvalue # 1.0
df # 1 (= 1*1)

print(evalue)
'''
[[59.84016393 97.15983607]
 [33.15983607 53.84016393]]
'''