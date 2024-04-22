'''
step02_DecisionTree_parameter.py

의사결정트리 주요 Hyper parameter 
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# tree 시각화 
from sklearn.tree import export_graphviz
from graphviz import Source  


iris = load_iris()

x_names = iris.feature_names # x변수 이름 
labels = iris.target_names # ['setosa', 'versicolor', 'virginica']

X = iris.data
y = iris.target
X
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


############################
### Hyper parameter 
############################
'''
criterion = 'gini' : 중요변수 선정 기준, 
 -> criterion : {"gini", "entropy"}, default="gini"
splitter = 'best' : 각 노드에서 분할을 선택하는 데 사용되는 전략, 
max_depth = None : tree 최대 깊이, 
 -> max_depth : int, default=None
 -> max_depth=None : min_samples_split의 샘플 수 보다 적을 때 까지 tree 깊이 생성 
 -> 과적합 제어 역할 : 값이 클 수록 과대적합, 적을 수록 과소적합 
min_samples_split = 2 : 내부 노드를 분할하는 데 필요한 최소 샘플 수(기본 2개)
 -> int or float, default=2    
 -> 과적합 제어 역할 : 값이 클 수록 과소적합, 적을 수록 과대적합 
'''

# model : default parameter
model = DecisionTreeClassifier(criterion='gini',
                               random_state=123, 
                               max_depth=None,
                               min_samples_split=2)

model.fit(X=X_train, y=y_train) # model 학습 

dir(model)
'''
feature_importances_ : X변수의 중요도
max_depth  : 트리 최대 깊이
get_depth()  : 트리 깊이 
max_features_: X변수 최대 길이
score() 
'''
model.get_depth()  # 5
model.max_features_ # 4
model.max_depth # None ; 파라미터로 사용됨

# model 평가 : 과적합(overfitting) 유무 확인  
model.score(X=X_train, y=y_train) # 1.0
model.score(X=X_test, y=y_test)   # 0.95555
# 과적합 우려됨


# tree 시각화 
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names=x_names,
                class_names=labels,
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 
'''
분류결과 해석 
setosa : 꽃잎길이(petal length)가 짧은 경우
virginica : 꽃잎너비가 넓고, 꽃잎길이가 긴 경우 
versicolor: 꽃잎길이가 중간이고 (petal length <=5.35), 꽃잎 넓이가 중간정도 큰 경우????
'''
####################
#### 과적합 방지를 위한 파라미터 설정(트리 깊이 지정)
####################
model_1 = DecisionTreeClassifier(criterion='entropy',
                               random_state=123, 
                               max_depth=3, # 4, 5 번째 가지치기
                               min_samples_split=2)

model_1.fit(X=X_train, y=y_train) # model 학습 

model_1.score(X=X_train, y=y_train) # 0.980952380
model_1.score(X=X_test, y=y_test)   # 0.93333333

# tree 시각화
graph = export_graphviz(model_1,
                out_file="tree_graph.dot",
                feature_names=x_names,
                class_names=labels,
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 
