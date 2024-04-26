'''
영화 추천 시스템 알고리즘
 - 추천 대상자 : Toby   
 - 유사도 평점 = 미관람 영화평점 * Toby와의 유사도
 - 추천 영화 예측 = 유사도 평점 / Toby와의 유사도
'''

import pandas as pd

# 데이터 가져오기 
ratings = pd.read_csv(r'C:\ITWILL\4_Python_ML\data\movie_rating.csv')
print(ratings) 
'''
     critic      title  rating
0      Jack       Lady     3.0
1      Jack     Snakes     4.0
2      Jack     You Me     3.5
3      Jack   Superman     5.0 ...
'''

### 1. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
movie_ratings = pd.pivot_table(ratings,
               index = 'title', # 행으로 들어갈 칼럼
               columns = 'critic', # 열로 들어갈 칼럼 
               values = 'rating').reset_index()
# reset_index() 하지 않으면 첫번째 칼럼이 index로 들어감 

print(movie_ratings)  
'''
<< movie rating >>

critic      title  Claudia  Gene  Jack  Lisa  Mick  Toby
0         Just My      3.0   1.5   NaN   3.0   2.0   NaN
1            Lady      NaN   3.0   3.0   2.5   3.0   NaN
2          Snakes      3.5   3.5   4.0   3.5   4.0   4.5
3        Superman      4.0   5.0   5.0   3.5   3.0   4.0
4       The Night      4.5   3.0   3.0   3.0   3.0   NaN
5          You Me      2.5   3.5   3.5   2.5   2.0   1.0
'''


### 2. 사용자 유사도 계산(상관계수 R)  
'''
< UBCF of Recommend system 원리>
결측치(NA) : 아직 정보가 없음 => 추천 시스템으로 추천 해준다. 
결측치가 아닌 정보들에 대해 유저 간 상관관계 계산을 한다. 
상관관계가 높은 사용자를 기반으로 NA를 유추해 추천할 수 있다.

결측이 많아 정확한 추천이 어려운 경우 => 특이값 분해 (SVD)

'''
sim_users = movie_ratings.iloc[:,1:].corr().reset_index() 
# corr(method='pearson')
print(sim_users) 
'''
critic   critic   Claudia      Gene      Jack      Lisa      Mick      Toby
0       Claudia  1.000000  0.314970  0.028571  0.566947  0.566947  0.893405
1          Gene  0.314970  1.000000  0.963796  0.396059  0.411765  0.381246
2          Jack  0.028571  0.963796  1.000000  0.747018  0.211289  0.662849
3          Lisa  0.566947  0.396059  0.747018  1.000000  0.594089  0.991241
4          Mick  0.566947  0.411765  0.211289  0.594089  1.000000  0.924473
5          Toby  0.893405  0.381246  0.662849  0.991241  0.924473  1.000000
'''


# Toby와의 상관계수 
sim_users['Toby'] # Toby vs others
'''
0    0.893405
1    0.381246
2    0.662849
3    0.991241  => Lisa
4    0.924473  => Mick
5    1.000000
'''


### 3. Toby 미관람 영화 추출  
# 1) movie_ratings table(원본테이블)에서 title, Toby 칼럼으로 subset 작성 

toby_rating = movie_ratings[['title', 'Toby']]  
print(toby_rating)
'''
critic title     Toby
0    Just My     NaN
1       Lady     NaN
2     Snakes     4.5
3   Superman     4.0
4  The Night     NaN
5     You Me     1.0
'''

# 2) Toby 미관람 영화제목 추출 
# 형식) DF.칼럼[DF.칼럼.isnull()]
toby_not_see = toby_rating.title[toby_rating.Toby.isnull()] 
print(toby_not_see) # rating null 조건으로 title 추출 
'''
0      Just My
1         Lady
4    The Night
'''
type(toby_not_see)


# 3) raw data에서 Toby 미관람 영화만 subset 생성 
rating_t = ratings[ratings.title.isin(toby_not_see)] # 3편 영화제목 
# Just My, Lady, The Night 
print(rating_t)
# 나머지 비평가들의 평점 ★★★
'''
     critic      title  rating
0      Jack       Lady     3.0
4      Jack  The Night     3.0
5      Mick       Lady     3.0
:
30     Gene  The Night     3.0
'''


# 4. Toby 미관람 영화 + Toby 유사도 join
# 1) Toby 유사도 추출 
# sim_users : Toby에 대한 상관관계 
toby_sim = sim_users[['critic','Toby']] # critic vs Toby 유사도 
toby_sim
'''
critic   critic      Toby
0       Claudia  0.893405
1          Gene  0.381246
2          Jack  0.662849
3          Lisa  0.991241
4          Mick  0.924473
5          Toby  1.000000
'''

# 2) 평가자 기준 병합  
rating_t = pd.merge(rating_t, toby_sim, on='critic')
print(rating_t)
'''
     critic      title  rating      Toby
0      Jack       Lady     3.0  0.662849
1      Jack  The Night     3.0  0.662849
2      Mick       Lady     3.0  0.924473
'''

print(3*0.662849)
### 5. 유사도 평점 계산 = Toby미관람 영화 평점 * Tody 유사도 
rating_t['sim_rating'] = rating_t.rating * rating_t.Toby
print(rating_t)
'''
     critic      title  rating      Toby  sim_rating
0      Jack       Lady     3.0    0.662849    1.988547 #3*0.662849
1      Jack  The Night     3.0    0.662849    1.988547
2      Mick       Lady     3.0    0.924473    2.773420
.....
[해설] Toby 미관람 영화평점과 Toby 유사도가 클 수록 유사도 평점은 커진다.
'''

### 6. 영화제목별 rating, Toby유사도, 유사도평점의 합계
group_sum = rating_t.groupby(['title']).sum() # 영화 제목별 합계
'''
           rating      Toby  sim_rating
title                                  
Just My       9.5  3.190366    8.074754
Lady         11.5  2.959810    8.383808
The Night    16.5  3.853215   12.899752
'''
 
### 7. Toby 영화추천 예측 = 유사도평점합계 / Tody유사도합계
print('\n*** 영화 추천 결과 ***')

# 시리즈로 나누기 ★★★
group_sum['predict'] = group_sum.sim_rating / group_sum.Toby
print(group_sum)
'''
           rating      Toby  sim_rating   predict
title                                            
Just My       9.5  3.190366    8.074754  2.530981
Lady         11.5  2.959810    8.383808  2.832550
The Night    16.5  3.853215   12.899752  3.347790 -> 추천영화 
'''
