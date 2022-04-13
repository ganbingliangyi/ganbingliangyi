#!/usr/bin/env python
# coding: utf-8

# # Pearson相关

# In[9]:


import numpy as np
import pandas as pd
from scipy import stats
filePath_01 = 'C:/Users/46685/Desktop/科研数据/新建处理后后.xlsx'
a = pd.read_excel(filePath_01,sheet_name = 'Sheet1')
data_pearson = a[["CA724","颈动脉平均值"]]
data_pearson.head()
stats.pearsonr(data_pearson.CA724, data_pearson.颈动脉平均值)


# In[10]:


## 通过相关系数绝对值范围来判断变量的相关强度：

## 0.8-1.0 极强相关
## 0.6-0.8 强相关
## 0.4-0.6 中等程度相关
## 0.2-0.4 弱相关
## 0.0-0.2 极弱相关或无相关


# In[ ]:




