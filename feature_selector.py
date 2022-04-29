#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font',family='Times New Roman')
a = 'C:/Users/46685/Desktop/科研数据/数据汇总/新建处理后后逻辑回归1.xlsx'
dataset= pd.read_excel(a,sheet_name = 'Sheet1')
dataset.head() #显示前几排数据
x=dataset.iloc[:,0:17]
y=dataset.iloc[:,17]
print(x)
print(y)


# In[32]:


from feature_selector import FeatureSelector
fs = FeatureSelector( data= x, labels = y)
fs.identify_missing(missing_threshold = 0.6)
fs.missing_stats.head()


# In[ ]:





# In[33]:


fs.identify_collinear(correlation_threshold=0.7, one_hot=False)
correlated_features = fs.ops['collinear']
print(correlated_features)


# In[34]:


fs.plot_collinear()


# In[35]:


# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification', 
 eval_metric = 'auc', 
 n_iterations = 10, 
 early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']


# In[36]:


# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 16)


# In[37]:


fs.identify_low_importance(cumulative_importance = 0.99)
fs.feature_importances.head(17)


# In[38]:


fs.identify_single_unique()
fs.plot_unique()


# In[ ]:




