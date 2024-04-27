#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font',family='Times New Roman')
a = 'C:/Users/46685/Desktop/论文王/4.4炎症和营养指标2.xlsx'
dataset= pd.read_excel(a,sheet_name = '4.4炎症和营养指标')
dataset.head() #显示前几排数据
x=dataset.iloc[:,0:15]
y=dataset.iloc[:,15]
print(x)
print(y)


# In[115]:


from sklearn.model_selection import train_test_split  #导入样本划分的库
### x_trainSVM,x_testSVM,y_trainSVM,y_testSVM=train_test_split(x,y,test_size=0.15,random_state=0) # 样本进行划分

x_trainRF,x_testRF,y_trainRF,y_testRF=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分 (随机森林)

x_trainLR,x_testLR,y_trainLR,y_testLR=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分 （逻辑回归）

x_trainDT,x_testDT,y_trainDT,y_testDT=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分 （决策树）

x_trainAB,x_testAB,y_trainAB,y_testAB=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分 （AdaBoost）

x_trainNB,x_testNB,y_trainNB,y_testNB=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分（贝叶斯）


# In[116]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


# In[117]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

LR.fit(x_trainLR,y_trainLR)
## 输出其在训练数据和验证数据集上的预测精度

y_predictLR = LR.predict(x_testLR)

print(classification_report(y_testLR,y_predictLR)) 


# In[118]:


from sklearn.naive_bayes import GaussianNB

# 创建高斯朴素贝叶斯模型
NB = GaussianNB()
 
# 训练模型
NB.fit(x_trainNB, y_trainNB)
 
y_predictNB = NB.predict(x_testNB)

print(classification_report(y_testNB,y_predictNB)) 




### from sklearn.svm import SVC

### model=SVC(C=0.1,kernel='rbf',degree=0.3,gamma='auto',probability=True,random_state=0, max_iter=1000)  #C越大代表惩罚程度越大，越不能容忍有点集交错的问题，但有可能会过拟合（defaul C=1）
### model.fit(x_trainSVM,y_trainSVM)         ##fit数,gamma='auto'据                                  #kernel常规的有‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ ，默认的是rbf；
### y_predictSVM = model.predict(x_testSVM)                                      

### print(classification_report(y_testSVM,y_predictSVM))        # 输出相关结果的函数


# In[119]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators = 100, # 树的数量
                              max_depth= 5,       # 子树最大深度
                              oob_score=True,
                              class_weight = "balanced",
                              random_state=1)
RF.fit(x_trainRF,y_trainRF)
## 输出其在训练数据和验证数据集上的预测精度

y_predictRF = RF.predict(x_testRF)

print(classification_report(y_testRF,y_predictRF)) 


# In[120]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
dtree.fit(x_trainDT,y_trainDT)
y_predictDT = dtree.predict(x_testDT)

from sklearn.tree import plot_tree
from sklearn.metrics import classification_report
print(classification_report(y_testDT,y_predictDT))    # 输出相关结果的函数


# In[121]:


from sklearn.ensemble import AdaBoostClassifier

dtc_cv = AdaBoostClassifier(learning_rate=0.0001,n_estimators=500,random_state=0)
dtc_cv.fit(x_trainAB,y_trainAB)
## 输出其在训练数据和验证数据集上的预测精度

y_predictAB = dtc_cv.predict(x_testAB)
print(classification_report(y_testAB,y_predictAB)) 


# In[122]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np


# In[123]:


### fprSVM,tprSVM,thresholdSVM = roc_curve(y_predictSVM,y_testSVM) ###计算真正率和假正率
### roc_aucSVM = auc(fprSVM,tprSVM) ###计算auc的值
### print(fprSVM.shape, tprSVM.shape, thresholdSVM.shape)


fprRF,tprRF,thresholdRF = roc_curve(y_predictRF,y_testRF) ###计算真正率和假正率
roc_aucRF = auc(fprRF,tprRF) ###计算auc的值
print(fprRF.shape, tprRF.shape, thresholdRF.shape)

fprLR,tprLR,thresholdLR = roc_curve(y_predictLR,y_testLR) ###计算真正率和假正率
roc_aucLR = auc(fprLR,tprLR) ###计算auc的值
print(fprLR.shape, tprLR.shape, thresholdLR.shape)

fprDT,tprDT,thresholdDT = roc_curve(y_predictDT,y_testDT) ###计算真正率和假正率
roc_aucDT = auc(fprDT,tprDT) ###计算auc的值
print(fprDT.shape, tprDT.shape, thresholdDT.shape)

fprAB,tprAB,thresholdAB = roc_curve(y_predictAB,y_testAB) ###计算真正率和假正率
roc_aucAB = auc(fprAB,tprAB) ###计算auc的值
print(fprAB.shape, tprAB.shape, thresholdAB.shape)

fprNB,tprNB,thresholdNB = roc_curve(y_predictNB,y_testNB) ###计算真正率和假正率
roc_aucNB = auc(fprNB,tprNB) ###计算auc的值
print(fprNB.shape, tprNB.shape, thresholdNB.shape)

plt.figure()
lw = 2
plt.figure (figsize=(5,5),dpi=1000)

### plt.plot(fprSVM, tprSVM, color='darkorange',
    ###    lw=lw, label='SVM (area = %0.2f)' % roc_aucSVM) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fprRF, tprRF, color='red',
          lw=lw,label='RandomForestClassifier (area = %0.2f)' % roc_aucRF) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fprLR, tprLR, color='green',
         lw=lw, label='LogisticRegression (area = %0.2f)' % roc_aucLR) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fprDT, tprDT, color='blue',
         lw=lw, label='DecisionTreeClassifier (area = %0.2f)' % roc_aucDT) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fprAB, tprAB, color='yellow',
         lw=lw, label='AdaBoostClassifier (area = %0.2f)' % roc_aucAB) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot(fprNB, tprNB, color='purple',
         lw=lw, label='GaussianNB (area = %0.2f)' % roc_aucNB) ###假正率为横坐标，真正率为纵坐标做曲线

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') #中间的虚线
plt.rc('font',family='Times New Roman')##字体
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




