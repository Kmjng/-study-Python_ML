'''
 문1) 신입사원 면접시험(interview.csv) 데이터 셋을 이용하여 다음과 같이 군집모델을 생성하시오.
 <조건1> 대상칼럼 : 가치관,전문지식,발표력,인성,창의력,자격증,종합점수 
 <조건2> 계층적 군집분석의 완전연결방식 적용 
 <조건3> 덴드로그램 시각화 : 군집 결과 확인   
'''

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # 계층적 군집 model
import matplotlib.pyplot as plt

# data loading - 신입사원 면접시험 데이터 셋 
interview = pd.read_csv("C:/ITWILL/4_Python_ML/data/interview.csv", encoding='ms949')
print(interview.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15 entries, 0 to 14
Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   no      15 non-null     int64 
 1   가치관     15 non-null     int64 
 2   전문지식    15 non-null     int64 
 3   발표력     15 non-null     int64 
 4   인성      15 non-null     int64 
 5   창의력     15 non-null     int64 
 6   자격증     15 non-null     int64 
 7   종합점수    15 non-null     int64 
 8   합격여부    15 non-null     object
dtypes: int64(8), object(1)
memory usage: 1.2+ KB
None

1 ~ 7 칼럼으로 군집화 
'''

# <조건1> subset 생성 : no, 합격여부 칼럼을 제외한 나머지 칼럼 
interview = interview.drop(['no','합격여부'], axis =1 )


# <조건2> 계층적 군집분석  : 군집화 방식 = 'complete' 
clusters = linkage(interview, method = 'complete')
clusters

# <조건3> 덴드로그램 시각화 : 군집 결과 확인
plt.figure(figsize=(40, 20))
dendrogram(clusters, 
           leaf_rotation=90,
           leaf_font_size=20,)
plt.show()


# <조건4> 군집 자르기 : 최대클러스터 개수 3개 지정  
cut_clusters = fcluster(clusters, t= 3, criterion ='maxclust')
cut_clusters 
#[2, 2, 1, 2, 1, 2, 3, 3, 1, 3, 1, 3, 2, 1, 3]

# <조건5> df에 cluster 칼럼 추가 & 군집별 특성 분석(그룹 통계 이용)
interview['cluster']=cut_clusters

group_interview = interview.groupby('cluster')
group_interview.mean().T
'''
cluster     1     2     3
가치관      11.0  19.0  14.4
전문지식     15.2  14.4  18.8
발표력      19.4  15.6  10.8
인성       11.0  14.8   9.4
창의력       6.2  11.8  18.2
자격증       0.4   1.0   0.0
종합점수     62.8  75.6  71.6
'''
interview[interview['cluster']==2]
'''
    가치관  전문지식  발표력  인성  창의력  자격증  종합점수  cluster
0    20    15   15  15   12    1    77        2
1    19    15   14  18   13    1    79        2
3    18    15   15  14   13    1    75        2
5    20    13   18  15   11    1    77        2
12   18    14   16  12   10    1    70        2
'''

