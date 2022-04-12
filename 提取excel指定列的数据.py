# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 22:11:55 2022

@author: 46685
"""

import pandas as pd
filePath_01 = 'C:/Users/46685/Desktop/科研数据/新建处理后后.xlsx'
a= pd.read_excel(filePath_01,sheet_name = 'Sheet1')
b = a[["性别", "年龄", "收缩压", "舒张压", "体重指数", "CA724分组","总胆固醇"]]
b.to_excel("C:/Users/46685/Desktop/科研数据/新建处理后后后.xlsx")