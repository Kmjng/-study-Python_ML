# -*- coding: utf-8 -*-
"""
 차원의 의미 ★★★
 차원 : 독립적인 축이나 방향의 수 
 행렬 : 행축의 차원은 데이터 수 (관측치) 
 행렬 : 열축의 차원은 특징 수 (columns; feature; 독립변수)
 
 선형대수 관련 함수 
  - 수학의 한 분야 
  - 벡터 또는 행렬을 대상으로 한 연산
  
  정방행렬 생성 : np.arange(1,10).reshape(3,3)
  n차원 단위행렬: eye_mat = np.eye(n)
  대각성분 추출 : diag_vec = np.diag(x)
  대각합 : trace_scala = np.trace(x) 
  대각행렬 : (단위행렬)*(차원일치하는 원소리스트)
              diag_mat = eye_mat * diag_vec
  행렬식D : np.linalg.det(x)
  역행렬 : np.linalg.inv(x) 
  내적 : x2.dot(x1)

"""


import numpy as np 

# 1. 선형대수 관련 함수
 
# 1) 단위행렬 : 대각원소가 1이고, 나머지는 모두 0인 n차 정방행렬
# n차원 단위행렬 
# np.eye(n)
eye_mat = np.eye(3) 
print(eye_mat)


# 정방행렬 x
x = np.arange(1,10).reshape(3,3)
print(x)


# 2) 대각성분 추출 
diag_vec = np.diag(x)
print(diag_vec) # 대각성분 : [1 5 9]
# 쓰임: 분류정확도 계산 ★★★ 
diag_vec.shape # (3,)

# 3) 대각합 : 정방행렬의 대각에 위치한 원소들의 합 
trace_scala = np.trace(x) 
print(trace_scala)


# 4) 대각행렬 : 대각성분 이외의 모든 성분이 모두 '0'인 n차 정방행렬
diag_mat = eye_mat * diag_vec # 차원 맞아야 함 ★★★
print(diag_mat)


# 5) 행렬식(determinant) : 대각원소의 곱과 차 연산으로 scala 반환 
# 행렬식 D(A) = ad - bc = 0 이면 역행렬 없음
# det(x)
'''
행렬식 용도 : 역행렬 존재 여부, 행렬식이 0이 아닌 경우 
            벡터들이 선형적으로 독립 ★★★★
'''

x = np.array([[3,4], [1,2]])
print(x)
'''
 v1 v2 (독립변수v1, v2)
[[3 4]
 [1 2]]
'''
det = np.linalg.det(x)
print(det) 

dir(np.linalg)
'''

det() : 행렬식
eig() : 고유값, 고유벡터 (차원축소)
inv() : 역행렬
multi_dot() : 행렬곱
norm() : 벡터 크기
solve() : 해 구하기 
svd() : 특이값으로 행렬분해

'''

################################
## 역행렬이 존재하지 않은 경우 
################################

# 1) 행렬 X  
x2 = np.array([[3,0], [1,0]])
print(x2)

# 2) 행렬식 
np.linalg.det(x2) # 0.0 : 행렬식 결과 = 0

# 3) 역행렬 : 특이 행렬 - 역행렬 존재 안함(error) 
np.linalg.inv(x2) # LinAlgError: Singular matrix




# 6) 역행렬(inverse matrix) : 행렬식의 역수와 정방행렬 곱 
'''
역행렬 용도 : 회귀 분석에서 최소 제곱 추정
    회귀방정식 :  Y = X * a(a; 회귀계수:가중치) ★★★
    최소 제곱 추정 : 손실를 최소화하는 회귀계수를 찾는 역할      
'''

inv_mat = np.linalg.inv(x)
print(inv_mat)
'''
[[ 1.  -2. ]
 [-0.5  1.5]]
'''

################################
## 회귀 분석에서 최소 제곱 추정
################################
'''
★★★★
- 정답 y 넣어서 회귀계수a를 예측한다.
- y_pred 에 위에서 구한 회귀계수를 활용 
'''

# 1) 데이터 생성
X = np.array([[1, 1], [1, 2], [1, 3]]) # 독립변수 [x1, x2]
#  주의 : 관측치(행) != 독립변수(열)
Y = np.array([2, 3, 4]) # 종속변수 (정답역할)
# 더하기 예측..
print(X)
'''
 x1 x2 
[[1 1]
 [1 2]
 [1 3]] => (3,2)

X.T => (2,3)
'''

# 2) 역행렬
XtX_inv = np.linalg.inv(X.T.dot(X)) # 역행렬 
X.T.dot(X) 
'''
X.T innerproduct X => (2,2)
array([[ 3,  6],
       [ 6, 14]])
'''

# 3) 행렬곱; inner product; dot()
XtY = X.T.dot(Y) # >> array([ 9, 20])


# 4) 최소 제곱 추정 계산
beta_hat = XtX_inv.dot(XtY)  

print("최소 제곱 추정값 :", beta_hat) 
# 최소 제곱 추정값 : [1. 1.] # 각 변수의 가중치a (회귀계수)

# 관측치1 [1 1]에 대한 예측
y_pred = (1*1) + (1*1) # (X1*a1)+(X2*a2) # 2 

# 관측치2 [1 2]에 대한 예측
y_pred_2 = (1*1) + (2*1) # (X1*a1)+(X2*a2) # 3

# 회귀계수; 기울기; 가중치 
a = np.array([[1],[1]])
a.shape # (2,1) 

# 독립변수 행렬 
X.shape #  (3,2)

y_pred = X.dot(a) # 선형회귀 공식 
y_pred
'''
array([[2],
       [3],
       [4]])
'''
