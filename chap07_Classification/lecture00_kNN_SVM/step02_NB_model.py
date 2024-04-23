'''
Naive Bayes 이론에 근거한 통계적 분류기

 1. GaussianNB  : x변수가 연속형이고, 정규분포인 경우 
 2. MultinomialNB : x변수가 단어 빈도수(텍스트 데이터)를 분류할 때 적합

 ** 사이킷런 데이터셋 기능
    subset='all'
    subset='train'
    subset='test' 
    (subset 파라미터 넣으면 알아서 쪼개줌)
    categories = '칼럼리스트명'
    (포함할 카테고리들(피쳐들))
'''

###############################
### news groups 분류 
###############################

#from sklearn.naive_bayes import GaussianNB # x변수가 연속형  
from sklearn.naive_bayes import MultinomialNB # tfidf 문서분류
from sklearn.datasets import fetch_20newsgroups # news 데이터셋 
from sklearn.feature_extraction.text import TfidfVectorizer# dtm(희소행렬) 
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가 


# 1. dataset 가져오기 (사이킷런 데이터셋)
newsgroups = fetch_20newsgroups(subset='all') # train/test load 

print(newsgroups.DESCR)
'''
**Data Set Characteristics:**

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========
'''
print(newsgroups.target_names) # 20개 뉴스 그룹 
len(newsgroups.target_names) # 20


# 2. train set 선택 : 4개 뉴스 그룹  
#cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
cats = newsgroups.target_names[:4]

news_train = fetch_20newsgroups(subset='train',categories=cats)
#  >> 훈련 데이터셋만 로드하며, 선택된 카테고리의 데이터만 포함
news_data = news_train.data # texts
len(news_data) # 2245
news_target = news_train.target # 0 ~ 3
news_target # [3, 2, 2, ..., 0, 1, 1] # 2245개

# 3. DTM(sparse matrix)
obj = TfidfVectorizer()
sparse_train = obj.fit_transform(news_data)
'''
<2245x62227 sparse matrix of type '<class 'numpy.float64'>'
	with 339686 stored elements in Compressed Sparse Row format>
'''    
sparse_train.shape # (2245, 62227) # 62227개로 단어 나뉜듯 ? 


# 4. NB 모델 생성 
nb = MultinomialNB() # alpha=.01 (default=1.0)
model = nb.fit(sparse_train, news_target) # 훈련셋 적용 


# 5. test dataset 4개 뉴스그룹 대상 : 희소행렬
news_test = fetch_20newsgroups(subset='test', categories=cats)
news_test_data = news_test.data # text 
len(news_test_data) # test 데이터 1,494개
y_true = news_test.target # 정답 데이터

# TfidfVectorizer()의 transform()메소드 ★★★
sparse_test = obj.transform(news_test_data)  
# 함수 주의 : fit_transform이 아님 ★★★
# (1494,~~) 행렬이 아닌, 기준문서인 obj객체의  (2245, 62227) 행렬에 맞춰 확인하기 위해 
# transform()메소드를 사용
sparse_test.shape # (1494, 62227)


# 6. model 평가 
y_pred = model.predict(sparse_test) # 예측치 

acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc) # accuracy = 0.852074

# 2) confusion matrix
con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
'''
    0     1   2   3
0 [[312   2   1   4]
1  [ 12 319  22  36]
2  [ 16  26 277  75]
3  [  1   8  18 365]]
'''

312 / (312 +  2  +  1 +  4) # 0.9780

