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


# In[4]:


from sklearn.model_selection import  train_test_split
from sklearn.ensemble import AdaBoostClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
dtc_cv = AdaBoostClassifier(learning_rate=0.0001,n_estimators=500,random_state=0)
dtc_cv.fit(x_train,y_train)
## 输出其在训练数据和验证数据集上的预测精度

y_predict = dtc_cv.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score


# In[5]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict)) 


# In[6]:


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




