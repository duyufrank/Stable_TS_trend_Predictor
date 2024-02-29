# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:55:32 2021

@author: Yu Du
"""
### This file only includes major functions, aiming to depict the workflow of this model
import pandas as pd
import numpy as np
from utils.macro_func_major import *
#%% module 1: data_preparation
#读取
skip = ['M5206730']
df = pd.read_csv('download_data5.csv',index_col = 0)
df_ref = pd.concat([pd.read_excel('宏观经济指标.xlsx','new'),pd.read_excel('宏观经济指标.xlsx','index')])
print('-----------------读取完成---------------\n')
#处理成月频、对社融部分数据进行调整
df_ref1 = ref_proc(df_ref.copy())
df1 = df2samefreq(df)
pos = df1['M5525755'].first_valid_index()
init = df1.loc[pos,'M5525755']*10000-df1.cumsum().loc[pos,'M5206730']
df1['M5206730']=((df1['M5206730'].cumsum()+init).dropna().pct_change(12)*100)
print('-----------------月频处理完成---------------\n')
#录入数据至df_ref
df2,df_ref2 = data_updater(df1,df_ref1.copy())
print('-----------------更新df_ref完成---------------\n')
#插值
df3,df_ref3= data_interpolate(df2,df_ref2.copy())
print('-----------------插值完成---------------\n')
#处理成同比
df4,df_ref4 = data_yoy(df3,df_ref3.copy(),skip=skip)
print('-----------------同比化完成---------------\n')
#标准化(方式：range)
df5,df_ref5 = data_standarize(df4,df_ref4.copy())
print('-----------------标准化完成---------------\n')
#季节性调整
x12_path = r'x13\x13as.exe'
spring_factor = generate_spring.compute_factor(pd.read_excel(r'春节.xlsx', sheet_name='Sheet1'), 13,6)
df6,df_ref6 = data_season_adj(df5,df_ref5.copy())
print('-----------------季节检验完成---------------\n')
#hp滤波
df7,df_ref7 = data_hp(df6,df_ref6.copy())
print('-----------------滤波完成---------------\n')
#输出
df_all = [df,df1,df2,df3,df4,df5,df6,df7]
df_ref_all = [df_ref,df_ref1,df_ref2,df_ref3,df_ref4,df_ref5,df_ref6,df_ref7]
output(df_all,df_ref_all)
df_ref7.to_csv('df_ref.csv')
df7.to_csv('df.csv')
print('-----------------输出完成---------------\n')
print('module1 done!\n\n\n')
#%% module : data_selection
#load
df = pd.read_csv('df.csv',index_col=0)
df_ref = pd.read_csv('df_ref.csv',index_col=0)
#filter by df_ref
df,df_ref = df_filter(df,df_ref)
print('----------初筛完成--------------\n')
#sort and offer a fixed mat  (cor_filter relies on the order of the mat)
df,df_ref = fix_mat(df,df_ref)
cor_mat = df.corr()
print('----------顺序调整完成--------------\n')
#filter by cor
dels = cor_filter(cor_mat)
print('----------成分筛选完成--------------\n')
print(f'删除{len(dels)}条数据')
#filter by cor and output
df = df[[i for i in df.columns if i not in dels]]
print(f'剩余{df.shape[1]}条数据')
df.to_csv('filtered_data.csv')
print('----------输出完成--------------\n')
print('module2 done!\n\n\n')
