# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:10:14 2021

@author: Yu Du
"""
import pandas as pd
from utils/macro_func_major import *
import time
code = 'M0000545'

start = time.time()
# load
df_ref = pd.read_csv('df_ref.csv',index_col=0)
df_comp = pd.read_csv('filtered_data.csv',index_col=0)
df = pd.read_csv('df.csv',index_col=0)
df_true = pd.read_excel('result_df.xlsx','季节性调整',index_col=0)
df_comp.index = pd.to_datetime(df_comp.index)
df.index = pd.to_datetime(df.index)
df.index.freq = 'M'
df_comp.index.freq = 'M'
# assign y
y = df[code]
y_true = df_true[code]
# filter self
if y.name in df_comp.columns:
    df_comp.drop(y.name,axis=1,inplace=True)
if y.name == 'M0001385':
    df_comp.drop('M0001384',axis=1,inplace=True)
end = time.time()
print('-------------------数据读取完成---------------------')
print(end-start)

# generate metrics table
start = time.time()
df_tp = df_comp.apply(lambda col:np.squeeze(get_indi(pd.DataFrame(col.dropna()))))
y_tp = get_indi(pd.DataFrame(y.dropna()))

df_result = pd.DataFrame(index=df_comp.columns)
df_result = gen_metrics(df_result,y,df_comp,y_tp,df_tp)
end = time.time()
print('-------------------数据打分完成---------------------')
print(end-start)

# compose and predict
pred = pd.DataFrame(index=range(1,13))
pred['pred'] = pred.index.map(lambda x:composed(y,df,df_result,x).diff().iat[-1])
pred['pred'] = pred['pred'].cumsum()+y_true.dropna().iat[-1]
