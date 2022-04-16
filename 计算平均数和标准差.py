#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
data = 'C:/Users/46685/Desktop/科研数据/新建处理后后.xlsx'
a= pd.read_excel(data,sheet_name = 'Sheet1')
b = a[["年龄","腰围"]]
print(b)
np.mean(b)   #计算平均数
np.std(b)   #计算标准差
np.std(b)**2    #计算方差





