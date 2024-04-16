'''
 연속형 변수 시각화 
 - 산점도, 산점도 행렬, boxplot 
'''

import matplotlib.pyplot as plt
import seaborn as sn

# seaborn 한글과 음수부호, 스타일 지원 
sn.set(font="Malgun Gothic", 
            rc={"axes.unicode_minus":False}, style="darkgrid")

# dataset load 
iris = sn.load_dataset('iris')
tips = sn.load_dataset('tips')


x = iris.sepal_length

# 1-1. displot : 히스토그램
sn.displot(data=iris, x='sepal_length', kind='hist')  
plt.title('iris Sepal length hist') # 단위 : Count 
plt.show()


# 1-2. displot : 밀도분포곡선 
# hue : 카테고리 값에 따라 다르게 시각화 ★★
sn.displot(data=iris, x='sepal_length', kind="kde", hue='species') 
plt.title('iris Sepal length kde') # 단위 : Density
plt.show()


# 2. 산점도 행렬(scatter matrix) 
# seaborn의 pairplot()  
sn.pairplot(data=iris, hue='species') 
plt.show()


# 3. 산점도 : 연속형+연속형+범주형(hue=집단변수)
# seaborn의 scatterplot()   
sn.scatterplot(x="sepal_length", y="petal_length", data=iris)
plt.title('산점도 (scatter)')
plt.show()

# 4. box-plot : 범주형+연속형 = 범주형
sn.boxplot(x='day', y ='total_bill', hue='sex', data =tips)
plt.title('성별 기준 요일별 지불금액')
plt.show() 