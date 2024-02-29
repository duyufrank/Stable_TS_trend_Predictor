# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:08:13 2021

@author: Yu Du
"""
import pandas as pd
from macro_func_base import *
#%% model1 data preparation
def ref_proc(df_ref):
    type2proc_type = {'cum_yoy':'yoy',
                     'yoy':'yoy',
                     'pct':'yoy',
                     'interest':'yoy',
                      'return':'yoy',
                     'index_diff':'yoy',
                     'index_cum':'index_cum',
                     'index':'index',
                     'unknown':'index'}
    df_ref['更新时间(年)'] = df_ref['更新时间'].apply(lambda x:str(x)[:4] if pd.notna(x) else '2021')
    df_ref['最近更新'] = df_ref.apply(lambda row: (row['更新时间(年)']=='2021')|
                              ((row['更新时间(年)']=='2020')&(row['频率']=='年')),axis=1)
    df_ref['type'] = df_ref.apply(lambda row:logic(row['指标名称']+str(row['Unnamed: 2'])) if pd.isna(row['数据类型']) else row['数据类型'],axis=1)
    df_ref['proc_type'] = df_ref['type'].apply(lambda x:type2proc_type[x])
    df_ref = df_ref.set_index('指标ID')
    return df_ref

def df2samefreq(df):
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()
    return df 

def data_updater(df,df_ref):    
    df_ref['期间最大空值数(月)'] = df_ref.apply(lambda row:max_nan_count(df[row.name]) if row['最近更新'] else np.nan,axis=1 )
    df_ref['指标持续时间(年)'] = df_ref.apply(lambda row:valid_period(df[row.name]) if row['最近更新'] else np.nan,axis=1 )
    return df,df_ref

def data_interpolate(df,df_ref):
    last_valid = df.apply(lambda col:col.last_valid_index())
    df_ref['method'] = df_ref['频率'].apply(lambda x:'pad' if (x=='季')|(x=='年') else 'linear')
    df = df.apply(lambda col:col.interpolate(method=df_ref.loc[col.name,'method']) if df_ref.at[col.name,'proc_type']!='index_cum' else col)
    df = df.apply(lambda col:del_col_tail(col,last_valid[col.name]))
    return df,df_ref

def data_yoy(df,df_ref,skip=[]):
    df = df.apply(lambda col:all2yoy(col,df_ref.at[col.name,'proc_type']) if col.name not in skip else col)
    return (df,df_ref)

def data_standarize(df,df_ref):
    df = df.apply(lambda col:standarize(col))
    return(df,df_ref)

def data_season_adj(df,df_ref):
    df_ref['season_test'] = df_ref.apply(lambda row:seasonality_test(df[row.name]) if row['最近更新'] else np.nan,axis=1)
    #df = df.apply(lambda col:x13(col.dropna()) if (df_ref.at[col.name,'season_test']<0.1)&(df_ref.at[col.name,'指标持续时间(年)']>3) else col)
    return(df,df_ref)

def data_hp(df,df_ref):
    df = df.apply(lambda col:hp_filter(hp_filter(col,14400)['irregular'],14)['trend'] if len(col.dropna())!=0 else np.nan)
    return (df,df_ref)

def output(df_all,df_ref_all):
    dfnames = ['原始数据','处理成月频','数据筛选','插值','处理成同比','标准化','季节性调整','hp过滤']
    df_refnames = ['原始数据','内部处理','数据筛选','插值','处理成同比','标准化','季节性调整','hp过滤']
    dictdf = {dfnames[i]:df_all[i] for i in range(len(df_all))}
    dictdf_ref = {df_refnames[i]:df_ref_all[i] for i in range(len(df_all))}
    
    writer = pd.ExcelWriter('result_df.xlsx')
    for name in dictdf:
        dictdf[name].to_excel(writer, name)
    writer.save()
    
    writer = pd.ExcelWriter('result_df_ref.xlsx')
    for name in dictdf_ref:
        dictdf_ref[name].to_excel(writer, name)
    writer.save()
 #%% model 2 X filter
def df_filter(df,df_ref,n=10):
   print('原数据有%d列'%len(df.columns))
   bool_cond = ((df_ref['最近更新'])&(df_ref['指标持续时间(年)']>n)&(df_ref['期间最大空值数(月)']<=13)&pd.notna(df_ref['season_test']))
   keep_list = df_ref[bool_cond].index.to_list()
   df_ref = df_ref[bool_cond]
   df = df[keep_list]
   print('现数据有%d列'%len(keep_list))
   return df,df_ref

def fix_mat(df,df_ref):
    fre_score_map = {'月':1, '季':0.9, '年':0.8, '周':1.1, '日':1.1}
    len_score = index2score(df_ref['指标名称'].apply(lambda x:len(x)),True,0)
    fre_score = df_ref['频率'].apply(lambda x:fre_score_map[x])
    long_score = index2score(df_ref['指标持续时间(年)'],0,0)
    nan_score = index2score(df_ref['期间最大空值数(月)'],1,0)
    df_ref['latest'] = df_ref.index.map(lambda x:pd.notna(df[x].iat[-1]))
    df_ref['final_score'] = len_score+fre_score+long_score+nan_score+3*df_ref['latest']
    df = df[df_ref.sort_values(by='final_score',ascending=False).index]
    return df,df_ref

def cor_filter(cormat):
    del_set = set()
    for index in cormat.index:
        if index not in del_set:
            items = cor_record(uptri(cormat).loc[index,:])
            del_set = del_set|items
    return del_set
#%% model 3 X2Y
def gen_metrics(df_result,y,df_comp,y_tp,df_tp):
    cor_abs_result = df_result.index.map(lambda x: compute_cor_abs(y,df_comp[x]))

    df_result['max_cor'] = df_result.index.map(lambda x: compute_cor(y,df_comp[x])[0])
    df_result['max_cor_abs'] = cor_abs_result.map(lambda x:x[0])
    df_result['cor_led'] = cor_abs_result.map(lambda x:x[1])
    
    kl_result = df_result.index.map(lambda x: compute_kl(y,df_comp[x],direction=df_result.loc[x,'max_cor']-df_result.loc[x,'max_cor_abs']))
    bb_result = df_result.index.map(lambda x: compute_BB(y_tp,df_tp[x],direction=df_result.loc[x,'max_cor']-df_result.loc[x,'max_cor_abs']))
    
    df_result['min_kl'] = kl_result.map(lambda x:x[0])
    df_result['kl_led'] = kl_result.map(lambda x:x[1])
    df_result['bb_match'] = bb_result.map(lambda x:x[0])
    df_result['bb_led'] = bb_result.map(lambda x:x[1])
    df_result['bb_led_std'] = bb_result.map(lambda x:x[2])
    df_result['bb_match2'] = bb_result.map(lambda x:x[3])
    df_result['num_bb_match'] = bb_result.map(lambda x:x[4])
    df_result['diff'] = np.abs(df_result['cor_led']-df_result['bb_led'])/3+np.abs(df_result['cor_led']-df_result['kl_led'])/3++np.abs(df_result['bb_led']-df_result['kl_led'])/3
    df_result['cor_score'] = index2score(df_result['max_cor_abs'],threshold = 0.5)
    df_result['kl_score'] = index2score(np.abs(df_result['min_kl']),True,threshold = 0.5)
    df_result['bb_match_score'] = index2score(df_result['bb_match'])
    df_result['bb_match_num_score'] = index2score(df_result['num_bb_match'])
    df_result['diff_score'] = index2score(df_result['diff'],True)
    df_result['lead_score'] = df_result['cor_led'].apply(lambda x:x>1).astype(int)
    df_result['final_score'] = df_result['cor_score']+df_result['bb_match_score']+df_result['bb_match_num_score']+df_result['diff_score']+0.5*df_result['kl_score']
    
    return df_result