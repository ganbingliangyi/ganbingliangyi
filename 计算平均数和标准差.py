#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
filePath_01 = 'C:/Users/46685/Desktop/科研数据/新建处理后后.xlsx'
a= pd.read_excel(filePath_01,sheet_name = 'Sheet1')
b = a[["年龄","腰围"]]
print(b)


# In[11]:


np.mean(b)   #计算平均数


# In[12]:


np.std(b)   #计算标准差


# In[13]:


np.std(b)**2    #计算方差


# In[ ]:




