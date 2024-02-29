# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:43:10 2021

@author: Yu Du
"""
import pandas as pd
from utils.macro_func_major import *
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
#filter by cor and output
df = df[[i for i in df.columns if i not in dels]]
df.to_csv('filtered_data.csv')
print('----------输出完成--------------\n')
