# -*- coding: utf-8 -*-
"""
SVD(Singular Value Decomposition) : 특이값을 이용한 행렬분해  
Numpy의 linalg 

A(M,N) = U @ D(대각행렬) @ V^T
구성요소 
▪ U행렬(Left Singular Vectors) : 
    원래 A행렬의 열 공간을 기반으로 한 직교 행렬(행의 특징) 
    # (M,k) ★

▪ D벡터(Singular Values): 
    원래 A행렬의 특이값 
    (대각행렬로 만들어서 사용한다) ;  np.diag(d)  
    # (k,k) ★

▪ V행렬(Right Singular Vectors): 
    원래 A행렬의 행 공간을 기반으로 한 직교 행렬(열의특징) 
    # (N,k)를 Transpose해서 (k,N)로 사용한다 ★

사용 함수 
np.linalg.svd(벡터) 
np.diag(특이값벡터).shape

"""

import numpy as np 

# 1. A행렬 만들기 
A = np.arange(1, 7).reshape(2,3)
print(A)
'''
[[1 2 3]
 [4 5 6]]
'''
# (2,3) ; user 2, item 3 


# 2. svd : 행렬분해 
svd = np.linalg.svd(A) 
svd
'''
행렬분해 하면 
(M*k) 
(1*k) 

(array([[-0.3863177 ,  0.92236578],  # U행렬 (M*k) # (2,2)
        [-0.92236578, -0.3863177 ]]),
 array([9.508032  , 0.77286964]),    # 특이값들로 구성된 벡터
                                     # (1,2) 이므로 (2,2)로 대각행렬해서 쓴다 
 array([[-0.42866713, -0.56630692, -0.7039467 ], # V행렬 (N*N)
        [-0.80596391, -0.11238241,  0.58119908],
        [ 0.40824829, -0.81649658,  0.40824829]]))
'''
u = svd[0] # u행렬  
d = svd[1] # 특이값들이 들어간 벡터  
v = svd[2] # v행렬  # (3,3)
np.diag(d).shape # (2,2) 
# np.diag ; d 벡터 원소를 갖고 대각행렬을 만듦 ★★★

X = u @ np.diag(d) @ v[:2]  # (2,2) @ (2,2) @ (2,3) = (2,3)
# v[:2] : V행렬 중 2행까지만 사용함 (행렬곱을 위해) ★
