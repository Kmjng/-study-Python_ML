# -*- coding: utf-8 -*-
"""
step03_logisticRegression_multiClass.py

 - 다항분류기(multi class classifier)  
"""
from sklearn.datasets import load_digits # dataset
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 


# 1. dataset loading 
digits = load_digits()
# 8*8의 64개의 픽셀을 갖고 있는 이미지데이터셋
image = digits.data # x변수 
label = digits.target # y변수 

image.shape  # (1797, 64) 
# 각 이미지는 0부터 15까지의 정수로 표현된 8x8 배열
# 밝기 정도가 정수로 표기(이산형 x변수)
label.shape  # (1797,) 

# 2. train_test_split
img_train, img_test, lab_train, lab_test = train_test_split(
                 image, label, 
                 test_size=0.3, 
                 random_state=123)


# 3. model 생성 
lr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   max_iter=100, 
                   multi_class='auto')
'''
multi_class='auto' : 다항분류(multinomial) ★★★
    penalty : {'L1','L2', 'elasticnet',None} , default = 'L2'
                -> 과적합 규제 : 'L1' - Lasso 회귀, 'L2' - ridge 회귀 
    C       : Cost 약자; 비용함수; float; default=1.0 
    
    random_state : int; Random State instance, default = None 
    solver :   최적화알고리즘; {'newton-cg','lbfgs','liblinear','sag','saga'} ; default = 'lbfgs'
    max_iter : 반복횟수 ; int, default = 100 

'''


model = lr.fit(X=img_train, y=lab_train)


# 4. model 평가 
y_pred = model.predict(img_test) # class 예측 

# 1) 혼동행렬(confusion matrix)
con_mat = confusion_matrix(lab_test, y_pred)
print(con_mat)
'''
[[59  0  0  0  0  0  0  0  0  0]
 [ 0 55  0  0  1  0  0  0  0  0]
 [ 0  0 53  0  0  0  0  0  0  0]
 [ 0  0  0 45  0  0  0  0  0  1]
 [ 0  0  0  0 60  0  0  1  0  0]
 [ 0  0  0  0  0 52  0  2  0  3]
 [ 0  1  0  0  0  1 55  0  0  0]
 [ 0  0  0  0  0  0  0 50  0  0]
 [ 0  4  0  0  1  0  0  0 43  0]
 [ 0  1  0  0  0  0  0  0  2 50]]
'''

# 2) 분류정확도(Accuracy)
accuracy = accuracy_score(lab_test, y_pred)
print('Accuracy =', accuracy)  
# Accuracy = 0.9666666666666667


# 3) heatmap 시각화 
import matplotlib.pyplot as plt
import seaborn as sn
  
# confusion matrix heatmap 
plt.figure(figsize=(6,6)) # size
sn.heatmap(con_mat, annot=True, fmt=".3f",
           linewidths=.5, square = True) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: ', format(accuracy,'.6f')
plt.title(all_sample_title, size = 18)
plt.show()

