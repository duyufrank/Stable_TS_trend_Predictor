# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:48:59 2021

@author: Yu Du
"""

import pandas as pd
import numpy as np 
from scipy import stats as stats
import statsmodels.api as sm
import generate_spring
from scipy.sparse import spdiags
from BB import *

#%% model1
# funcs in model1 -- clean and prepare data
def logic(name):
    if '累计同比' in name:
        dt_type = 'cum_yoy'
        return(dt_type)
    elif '同比' in name:
        dt_type = 'yoy'
        return(dt_type)
    elif '环比' in name:
        dt_type = 'pct'
        return(dt_type)
    elif '利率' in name:
        dt_type = 'interest'
        return(dt_type)
    elif '收益率' in name:
        dt_type = 'return'
        return(dt_type)
    elif '扩散指数' in name:
        dt_type = 'index_diff'
        return(dt_type)
    elif '累计' in name:
        dt_type = 'index_cum'
        return(dt_type)
    elif '指数' in name:
        dt_type = 'index' 
        return(dt_type)
    else:
        return('unknown')

def outlier_remove(series):
    if len(series[series.apply(lambda x:x<0)]) == 0:
        last = series.index[0]
    else:
        last = series[series.apply(lambda x:x<0)].index[-1]
    series = series[series.index.map(lambda x:x>last)]
    return series  
  
def max_nan_count(series):
        code = series.name
        df = pd.DataFrame(series)
        df['not NA'] = df[code].apply(pd.notna)
        df['cum']=df['not NA'].cumsum()
        df = df[df['cum']>0]
        last = df['cum'].max()
        df = df[df['cum']<last]
        return df.groupby('cum').count()['not NA'].max()-1
    
def valid_period(series):
    first = series.first_valid_index()
    last = series.last_valid_index()
    if first:
        return(last.year-first.year)
    else:
        return(np.nan)

def del_col_tail(series,last_index):
    name = series.name
    series = pd.DataFrame(series)
    series = series.apply(lambda row:np.nan if row.name>last_index else row[name],axis=1)
    return(series)

def all2yoy(series,dtype):
    def index_cum2index(series):
        series=series.dropna()
        temp = pd.DataFrame()
        for i in range(len(series)):
            if series.index[i].month == 1:
                if pd.isna(series.iloc[i]):
                    temp = temp.append(pd.DataFrame([series.iloc[i+1]/2], index=[series.index[i]]))
                else:
                    temp = temp.append(pd.DataFrame([series.iloc[i]],index=[series.index[i]]))
            elif series.index[i].month == 2:
                temp = temp.append(pd.DataFrame([series.iloc[i]/2], index=[series.index[i]]))
            else:
                temp = temp.append(pd.DataFrame([series.iloc[i]-series.iloc[i-1]],index=[series.index[i]]))
        return temp.squeeze()
        
    if dtype == 'yoy':
        return(series)
    if dtype == 'pct':
        series = series/100+1
        series = series.cumprod()
        series = series.pct_change(12,fill_method = None)
        return(series)
    if dtype == 'index_cum':
        series = index_cum2index(series)
        series = series.pct_change(12,fill_method = None)
        return(series)
    if dtype == 'index':
        series = series.pct_change(12,fill_method = None)
        return(series)
      
############not used yet########################
def index2yoy(series):
    series = series.pct_change(12).apply(lambda x:np.nan if np.isinf(x) else x)
    return series.fillna(method="ffill")
################################################
def standarize(series):
    max_num = np.nanmax(series)
    min_num = np.nanmin(series)
    mean = np.nanmean(series)
    series = series.apply(lambda x:(x-mean)/(max_num-min_num))
    return(series)

def seasonality_test(series):
    ''''''
    def series2mat(series):
        '''
        translate a series to a year-by-month matrix,
        only take series whose frequency is "month"
        '''
        name = series.name
        tempdf=pd.DataFrame(series)
        #generate two variable as index and cols
        tempdf['Y'] = tempdf.index.map(lambda x:x.year)
        tempdf['M'] = tempdf.index.map(lambda x:x.month)        
        return(tempdf.pivot(index='Y',columns='M',values=name))
    
    def f_test(mat):
        '''
        F-test. H0:means of every cols are equal.
        if p-value< alpha, a significance level, we can reject H0.
        Parameters
        ----------
        mat : a dataframe
        Returns
        -------
        f : F static
        p : p_value 
    
        '''
        col_num = mat.shape[1]#k
        #clean series without np.nan
        groups = [mat.iloc[:,i].dropna() for i in range(col_num)]
        #SSB
        counter = np.array([np.size(group) for group in groups])
        mean_groups = mat.sum().sum()/counter.sum()
        square = np.square(mat.mean()-mean_groups)
        ssb = (counter*square).sum()
        #MSB
        df1 = col_num-1#degree of freedom, not dataframe
        msb = ssb/df1
        #SSW
        counter2 = counter-1
        var = mat.var()
        ssw = (counter2*var).sum()
        #MSW
        df2 = sum(counter)-col_num
        msw = ssw/df2
        
        f = msb/msw
        p = 1-stats.f.cdf(f, df1, df2)
        return f,p
    mat = series2mat(series)
    f,p = f_test(mat)
    return p

def x13(Y):  # Y是一个Index是时间的Dataframe
    try:
        result = sm.tsa.x13_arima_analysis(Y, x12path=x12_path, freq='M', outlier=True, trading=True,
                                           prefer_x13=True, exog=spring_factor)
    except:
        result = sm.tsa.x13_arima_analysis(Y, x12path=x12_path, freq='M', outlier=False, trading=True,
                                           prefer_x13=True, exog=spring_factor)
    return result.trend


def hp_filter(Y1, smoothing, **kw):  
    Y1 = pd.DataFrame(Y1)# Y是一个Index是时间的Dataframe
    Y1 = Y1.dropna()
    Y = Y1.values
    numObs = Y.shape[0]
    e = np.tile(np.array([smoothing, -4 * smoothing, (1 + 6 * smoothing), - 4 * smoothing, smoothing]), (numObs, 1))
    A = spdiags(e.T, np.array([-2, -1, 0, 1, 2]), numObs, numObs).toarray()
    A[0, 0] = 1 + smoothing
    A[0, 1] = -2 * smoothing
    A[1, 0] = -2 * smoothing
    A[1, 1] = 1 + 5 * smoothing
    A[numObs - 2, numObs - 2] = 1 + 5 * smoothing
    A[numObs - 2, numObs - 1] = -2 * smoothing
    A[numObs - 1, numObs - 2] = -2 * smoothing
    A[numObs - 1, numObs - 1] = 1 + smoothing
    Trend = np.linalg.inv(A) @ Y
    Cyclical = Y - Trend
    return pd.DataFrame(np.squeeze(np.array([Trend, Cyclical]).T),
                        index=Y1.index, columns=['trend', 'irregular'])
#%% model2 
#funcs in model 2 -- select Xs
def index2score(series,inverse = False,threshold = 0.4):
    if inverse:
        maximum = np.nanmax(series)
        minimum = np.nanmin(series)
        series = series.fillna(maximum)
        series = series.apply(lambda x: abs(x-maximum))
        scores = series.apply(lambda x: max(x-threshold*maximum,0)/((1-threshold)*maximum))
        return scores
    else:
        maximum = np.nanmax(series)
        minimum = np.nanmin(series)
        series = series.fillna(0)
        scores = series.apply(lambda x: max(x-threshold*maximum,0)/((1-threshold)*maximum))
        return scores
    
def cor_record(series,threshold = 0.85):
    return set(series[series>threshold].index.tolist())

def diff(x,y):
    return(max(len(x-y),len(y-x)))

def uptri(cor_mat):
    mat = cor_mat.to_numpy()
    ndim = cor_mat.to_numpy().shape[0]
    for i in range(ndim):
        for j in range(i+1):
            mat[i][j] = 0
    return pd.DataFrame(mat,index = cor_mat.index, columns = cor_mat.columns)

#%% model3
# funcs in model 3 -- X2Y
def accord2y(y,series):
    range1 = np.nanmax(series)-np.nanmin(series)
    range2 = np.nanmax(y)-np.nanmin(y)
    series = range2/range1*series
    return series


def compute_cor(y,series):
    cor_result = {i:0 for i in range(1,13)}
    for i in cor_result:
        cor_result[i] = y.corr(series.shift(i))
    return max(cor_result.values()),max(cor_result,key=cor_result.get)

def compute_cor_abs(y,series):
    cor_result = {i:0 for i in range(1,13)}
    for i in cor_result:
        cor_result[i] = y.corr(series.shift(i))
    cor_result = {i:abs(cor_result[i]) for i in cor_result}
    return max(cor_result.values()),max(cor_result,key=cor_result.get)

def KL(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def compute_kl(y,series,direction=1):
    if direction<0:
        series = - series
    kl_result = {i:0 for i in range(1,13)}
    '''处理负值,exp'''
    y = y-y.min()+0.1
    series = series-series.min()+0.1
    for i in kl_result:
        '''强行弄出一个df是为了处理p,q使得p,q长度一致'''
        tempdf = pd.concat([y,series.shift(i)],axis=1).dropna()
        p = tempdf[y.name]
        q = tempdf[series.name]
        kl_result[i] = KL(p, q)
    return min(kl_result.values()),min(kl_result,key=kl_result.get)        

def compute_BB(y_tp,series_tp,direction=1):
    if direction<0:
        series_tp = -series_tp
    y_tp = np.squeeze(y_tp)
    tempdf = pd.concat([y_tp,series_tp],axis=1).dropna()
    y_tp = pd.DataFrame(tempdf.iloc[:,0])
    series_tp = pd.DataFrame(tempdf.iloc[:,1])
    extOrd, time, missing, missingEarly, extra = matchTurningPoints(y_tp, series_tp, lagFrom=0,lagTo=13, printDetails=False)
    match = pd.notna(extOrd).sum().sum()
    match_rate = match/(np.abs(series_tp).sum().sum())
    match_rate2 = match/(np.abs(y_tp).sum().sum())
    led_ave = time[pd.notna(time)].mean().mean()
    led_std = np.squeeze(time).dropna().to_numpy().std()
    return match_rate,led_ave,led_std,match_rate2,match#,time,extOrd#extOrd, timei, missing, missingEarly, extra

def test_composed(df,df_result,lead_rank,num_comps=30,auto=True):
    if auto & (num_comps>10):
        if (lead_rank>5):
            num_comps = 11
    lead_rank = lead_rank-1
    selected = df_result.sort_values(by='final_score',ascending = False)[df_result.sort_values(by='final_score',ascending = False)['cor_led']>lead_rank].head(num_comps)
    min_shift = selected['cor_led'].min()
    shift = selected['cor_led'].apply(lambda x:x-min_shift)
    df_selected = df[selected.index]
    df_selected = df_selected.apply(lambda col:col.shift(shift.at[col.name]))
    
    return df_selected

def last_order_check(df):
    check = df.iloc[-1,:]
    return(check[check.apply(lambda x:pd.isna(x))].index)

def composed(y,df,df_result,lead_rank,num_comps=30,auto=True,adjust=True):
    if auto & (num_comps>10):
        if (lead_rank>5):
            num_comps = 11
    lead_rank = lead_rank-1
    selected = df_result.sort_values(by='final_score',ascending = False)[df_result.sort_values(by='final_score',ascending = False)['cor_led']>lead_rank].head(num_comps)
    min_shift = selected['cor_led'].min()
    shift = selected['cor_led'].apply(lambda x:x-min_shift)
    select_codes = selected.index
    if adjust:
        select_codes = [i for i in select_codes if i not in last_order_check(test_composed(df,df_result,lead_rank+1,num_comps=30,auto=True))]
    df_selected = df[select_codes].apply(lambda col:-col if selected.loc[col.name,'max_cor']!= selected.loc[col.name,'max_cor_abs']else col)
    df_selected = df_selected.apply(lambda col:col.shift(shift.at[col.name])).fillna(0).to_numpy()
    rank = selected['final_score'].rank(ascending=False)
    #weight = 1/(rank.to_numpy()+1)
    weight = selected[selected.index.map(lambda x:x in select_codes)]['final_score'].to_numpy()
    #weight = np.array(num_comps*[1/num_comps])
    index =  pd.DataFrame(df_selected@(weight.T),index = df.index)
    index = index[index.index<=y.last_valid_index()]
    index = np.squeeze(index)
    index.name = f'composed_index_lead_{lead_rank+1}'
    return accord2y(y,standarize(index))

def accuracy(y,index,n):    
    x=pd.concat([np.sign(y.dropna().diff()),np.sign(index.shift(n).dropna().diff())],axis=1).dropna()
    return((x.iloc[:,1]==x.iloc[:,0]).sum()/len(x))

def tail_acc(y_true,index,n,threshold = 0.3):
    a = y_true.dropna().diff()
    up_thre = a.quantile(1-threshold)
    low_thre = a.quantile(threshold)
    a = a.apply(lambda x: x if ((x>up_thre) or (x<low_thre)) else np.nan).dropna()
    b = composed(n).shift(n).dropna().diff()
    x = pd.concat([np.sign(a),np.sign(b)],axis=1).dropna()
    return((x.iloc[:,1]==x.iloc[:,0]).sum()/len(x))

def pred_mse(y_true,n):
    y_true = y_true.diff()
    y_pred = composed(n).shift(n).diff()
    x = pd.concat([y_true,y_pred],axis=1).dropna()
    y_true = x.iloc[:,0]
    y_pred = x.iloc[:,1]
    return(mse(y_true,y_pred))      

def show_plots(y_true,n):
    y_true = y_true.diff()
    y_pred = composed(n).shift(n).diff()
    x = pd.concat([y_true,y_pred],axis=1).dropna()
    return x.iloc[:,0]-x.iloc[:,1] 

def nums_test(lead):
    df = pd.DataFrame(index=range(5,50))
    df['acc'] = df.index.map(lambda x:accuracy(y,composed(lead,x,0,0),lead))
    #df['tail'] = df.index.map(lambda x:tail_acc(y,composed(lead,x,0,0),lead))
    #df['tail_true'] = df.index.map(lambda x:tail_acc(y_true,composed(lead,x,0,0),lead))
    temp = df_result.sort_values(by='final_score',ascending = False)[df_result.sort_values(by='final_score',ascending = False)['cor_led']>=lead]
    df['lowest_score'] = df.index.map(lambda x:temp.head(x)['final_score'].iat[-1] if temp.shape[0]>=x else np.nan)
    return(df)



