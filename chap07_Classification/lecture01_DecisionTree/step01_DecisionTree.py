'''
step01_DecisionTree.py

Decision Tree 모델 & 시각화 
"""

'''
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier # 의사결정트리 모델  
from sklearn.metrics import accuracy_score # model 평가 

# tree 시각화 
from sklearn.tree import plot_tree, export_graphviz
from graphviz import Source # 설치 필요 : pip install graphviz

# 1. dataset load 
path = r'c:\ITWILL\4_Python_ML\data'
dataset = pd.read_csv(path + "/tree_data.csv")
print(dataset.info())
'''
iq         6 non-null int64 - iq수치
age        6 non-null int64 - 나이
income     6 non-null int64 - 수입
owner      6 non-null int64 - 사업가 유무 (0,1) -명목
unidegree  6 non-null int64 - 학위 유무   (0,1) -명목
smoking    6 non-null int64 - 흡연 유무 - y변수 
'''

# 2. 변수 선택 
cols = list(dataset.columns)
X = dataset[cols[:-1]]
y = dataset[cols[-1]]

# 3. model & 평가 
model = DecisionTreeClassifier(random_state=123).fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(acc)  # 1.0


# 4. tree 시각화 
feature_names = cols[:-1]  # x변수 이름 

# 의사결정트리 시각화 ★★★
plot_tree(model, feature_names = feature_names)  

'''
가장 잘 분류하는 x변수 : income (root node) 
root 조건을 만족(yes) - 불만족(no) 
지니불순도(gini)가 0이면 지니계수가 높음  
'''


# y변수 class 이름 
class_names = ['no', 'yes'] 


# 의사결정트리 파일 저장 & 콘솔 시각화 ★★★
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names = feature_names, # X변수 이름
                class_names = class_names,
                rounded=False,
                impurity=True, # 지니불순도 #False - 엔트로피
                filled=False) # filled = True : 색상 추가

#  file load 
file = open("tree_graph.dot", mode = 'r') # 현재폴더 내 파일
dot_graph = file.read()
  
Source(dot_graph) # tree 시각화 : Console 출력
# 오류발생시, 재시작

'''
분류결과 해석 
비흡연자(class= no) : 수입 적음 (income <= 24.0 ) 
흡연자(class= yes) : 수입 많고, 지능지수가 높음 (iq >= 105.0 )
지능지수 : 105 이하 - 상대적 수입이 작으면 흡연자 (income <= 43.0 ) 
                    - 상대적 수입이 높으면 비흡연자
'''
