import pandas as pd
import numpy as np
from macro_func_major import *
import warnings
import time as t
warnings.filterwarnings("ignore")

code = 'M0001227'
df_start = pd.read_csv('download_data4.csv',index_col = 0)
df_start = df_start.sort_index()
df_ref_start = pd.concat([pd.read_excel('宏观经济指标.xlsx','new'),pd.read_excel('宏观经济指标.xlsx','index')])
start_time = '2013-09-01'
end_time = '2021-10-31'
test_periods = pd.date_range(start=start_time, end=end_time, freq='M')
test_periods = [i.strftime("%Y-%m-%d") for i in test_periods]
result = dict()
x12_path = r'x13\x13as.exe'
spring_factor = generate_spring.compute_factor(pd.read_excel(r'春节.xlsx', sheet_name='Sheet1'), 13,6)
#result: {time:pred,range of y,range of y_true, components}
i=0
for time in test_periods:
    t_start = t.time()
    i+=1
    print(f'-----{i}/{len(test_periods)}-----')
    df = df_start[df_start.index.map(lambda x:x<=time)]
    df_ref = df_ref_start
    #处理成月频、对社融部分数据进行调整
    df_ref1 = ref_proc(df_ref.copy())
    df1 = df2samefreq(df)
    df1 = df1.apply(lambda x:outlier_remove(x) if '社会融' in df_ref1.loc[x.name,'指标名称'] else x)  
    #录入数据至df_ref
    df2,df_ref2 = data_updater(df1,df_ref1.copy())    
    #插值
    df3,df_ref3= data_interpolate(df2,df_ref2.copy())    
    #处理成同比
    df4,df_ref4 = data_yoy(df3,df_ref3.copy())    
    #标准化(方式：range)
    df5,df_ref5 = data_standarize(df4,df_ref4.copy())    
    #季节性调整
    df6,df_ref6 = data_season_adj(df5,df_ref5.copy())   
    #hp滤波
    df7,df_ref7 = data_hp(df6,df_ref6.copy())    
    #输出
    df_all = [df,df1,df2,df3,df4,df5,df6,df7]
    df_ref_all = [df_ref,df_ref1,df_ref2,df_ref3,df_ref4,df_ref5,df_ref6,df_ref7]   
    #load
    df = df7
    df_ref = df_ref7
    #filter by df_ref
    df,df_ref = df_filter(df,df_ref)  
    n1=df.shape[1]
    #sort and offer a fixed mat  (cor_filter relies on the order of the mat)
    df,df_ref = fix_mat(df,df_ref)
    cor_mat = df.corr()   
    #filter by cor
    dels = cor_filter(cor_mat)    
    #filter by cor and output
    df = df[[i for i in df.columns if i not in dels]]
    n2=df.shape[1]
    print('...')
    # load
    df_ref = df_ref7
    df_comp = df
    df = df7
    df_true = df4
    df_comp.index = pd.to_datetime(df_comp.index)
    df.index = pd.to_datetime(df.index)
    df.index.freq = 'M'
    df_comp.index.freq = 'M'
    # assign y
    y = df[code]
    y_true = df_true[code]
    y_range = np.nanmax(y.dropna())-np.nanmin(y.dropna())
    y_true_range = np.nanmax(y_true.dropna())-np.nanmin(y_true.dropna())
    # filter self
    if y.name in df_comp.columns:
        df_comp.drop(y.name,axis=1,inplace=True)
    if y.name == 'M0001385':
        df_comp.drop('M0001384',axis=1,inplace=True)
    # generate metrics table
    df_tp = df_comp.apply(lambda col:np.squeeze(get_indi(pd.DataFrame(col.dropna()))))
    y_tp = get_indi(pd.DataFrame(y.dropna()))
    
    df_result = pd.DataFrame(index=df_comp.columns)
    df_result = gen_metrics(df_result,y,df_comp,y_tp,df_tp)
    components = df_result.sort_values(by='final_score',ascending = False)['final_score'].to_dict()
  
    # compose and predict
    pred = pd.Series(index=range(1,13))
    pred = pd.Series(pred.index.map(lambda x:composed(y,df,df_result,x).diff().iat[-1]))
    pred_cum = pred.cumsum()
    result.update({time:[n1,n2,y_range,y_true_range,components,pred,pred_cum]})
    t_end = t.time()
    print('...')
    print(round(t_end-t_start,2))
x = pd.DataFrame(result).T
x.columns = ['初筛结果','X数量','y_range','y_true_range','components','pred','pred_cum']
x.to_csv(f'huice_{code}2.csv')    