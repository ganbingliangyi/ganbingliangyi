#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# 修复图片中文显示乱码及刻度显示缺失问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

get_ipython().run_line_magic('matplotlib', 'inline')

filePath_01 = 'C:/Users/46685/Desktop/科研数据/新建处理后后.xlsx'
a= pd.read_excel(filePath_01,sheet_name = 'Sheet1')
ls1 = a["baPWV平均值"]
ls2 = a["年龄"]
data = pd.DataFrame({'baPWV平均值':ls1,'年龄':ls2 })
# 首先绘制出各属性关系图
sns.pairplot(data,kind='scatter',diag_kind='kde')
for column in data.columns:
    u = data[column].mean() # 计算均值
    std = data[column].std() # 计算标准差
    r,p = scipy.stats.kstest(data[column],'norm',(u,std))
    if p>0.05:
        print('拒绝原假设，显著性水平为{}，变量{}服从正态分布'.format(p,column))
    else:
        print('接受原假设，显著性水平为{}，变量{}不服从正态分布'.format(p,column))


# In[ ]:




