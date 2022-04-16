#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']#让中文的地方显示出来
data = 'C:/Users/46685/Desktop/科研数据/提取数据/新建处理后后后后.xlsx'
df = pd.read_excel(data,sheet_name = 'Sheet1')
df.head()
df_coor=df.corr()
df_coor.head()
plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色

fig=sns.heatmap(df_coor,annot=True, vmax=1, square=True, cmap="Blues", fmt='.2f')#annot为热力图上显示数据；fmt='.2f'为数据保留小数点后两位,square呈现正方形，vmax最大值为1fig
fig.get_figure().savefig('df_corr.png',bbox_inches='tight',transparent=True)#保存图片
#bbox_inches让图片显示完整，transparent=True让图片背景透明


# In[ ]:




