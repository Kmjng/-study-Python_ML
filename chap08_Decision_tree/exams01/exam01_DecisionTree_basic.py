'''
 문1) load_breast_cancer 데이터 셋을 이용하여 다음과 같이 Decision Tree 모델을 생성하시오.
  <조건1> 75:25비율 train/test 데이터 셋 구성 
  <조건2> x변수 : cancer.data, y변수 : cancer.target
  <조건3> tree 최대 깊이 : 5 
  <조건4> decision tree 시각화 & 중요변수 확인
'''

from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
# tree 시각화 
from sklearn.tree import export_graphviz
from graphviz import Source 

# 데이터 셋 load 
cancer = load_breast_cancer()
# cancer_df = pd.DataFrame()
# <단계1> x변수 : cancer.data, y변수 : cancer.target
X = cancer.data
y = cancer.target 

X.shape # (569, 30) 
y.shape # (569,)

# <단계2> 75:25비율 train/test 데이터 셋 구성
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, 
                                                    random_state =123)


# <단계3> tree 최대 깊이 : 5
model = DecisionTreeClassifier(random_state=123, 
                               max_depth=5).fit(X_train,y_train)

model.score(X_test, y_test) # 0.972027972027972


# <단계4> decision tree 시각화 & 중요변수 확인 ★★★★
# cancer.target_names #['malignant', 'benign'] #악성, 양성

# model 평가 
model.score(X=X_test, y=y_test) # 0.972027972027972

x_imp = model.feature_importances_
print(x_imp) 
x_imp.shape #(30,) # x변수 30개 들에 대한 중요도
'''
[0.         0.0319083  0.00769218 0.         0.         0.00683749
 0.04425647 0.01162374 0.         0.         0.01017484 0.
 0.00996826 0.         0.         0.         0.00911666 0.
 0.         0.         0.69916976 0.04821522 0.         0.0173074
 0.         0.         0.         0.10372968 0.         0.        ]
'''
# 인덱스 정렬 
idx = x_imp.argsort()[::-1] # 여기서는 내림차순 해야 
idx[:5] #  [22, 27, 24, 21, 28] 
# 중요변수 5개 
cancer.feature_names[idx[:5]]
'''
['worst perimeter', 'worst concave points',
 'worst smoothness', 'worst texture', 'worst symmetry']
'''
# 外) 확인해볼 수 있다. 
cancer.feature_names # 30개 
'''
['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']
'''
cancer.feature_names[22] # 'worst perimeter'




# tree 시각화
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names=cancer.feature_names, # x변수 이름
                class_names=cancer.target_names, # y변수 레이블
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 


