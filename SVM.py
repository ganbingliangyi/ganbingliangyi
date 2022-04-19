#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font',family='Times New Roman')
a = 'C:/Users/46685/Desktop/科研数据/数据汇总/新建处理后后逻辑回归1.xlsx'
dataset= pd.read_excel(a,sheet_name = 'Sheet1')
dataset.head() #显示前几排数据
x=dataset.iloc[:,0:16]
y=dataset.iloc[:,17]
print(x)
print(y)


# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) # 样本进行划分
from sklearn.preprocessing import StandardScaler  ## 归一化
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
from sklearn.svm import SVC

model=SVC(C=1,kernel='rbf',degree=0.3,gamma='auto',probability=True,random_state=0, max_iter=1000)  #C越大代表惩罚程度越大，越不能容忍有点集交错的问题，但有可能会过拟合（defaul C=1）
model.fit(x_train,y_train)         ##fit数,gamma='auto'据                                  #kernel常规的有‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ ，默认的是rbf；
y_predict = model.predict(x_test)                                      


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))        # 输出相关结果的函数


# In[39]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
fpr,tpr,threshold = roc_curve(y_predict,y_test) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
print(fpr.shape, tpr.shape, threshold.shape)


plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
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




