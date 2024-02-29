import datetime
import pandas as pd
import numpy as np

def compute_factor(festival_date, past, future, start_date = None, end_date = None):
    if start_date and end_date:
        start_date = pd.date_range(start_date, periods=1, freq='M')[0]
        end_date = pd.date_range(end_date, periods=2, freq='M')[1]
    else:
        start_date = min(festival_date.values)[0]
        end_date = pd.date_range(max(festival_date.values)[0]+pd.Timedelta(days=1), periods=1,freq='M')[0]
    all_date2 = pd.DataFrame(0,index=pd.date_range(start_date, end_date, freq='M'), columns=['factor'])
    all_date = []
    for festival in festival_date.values:
        for i in range(-1*past, future+1):
            all_date.append(festival+pd.Timedelta(days=i))
    all_date = pd.DataFrame(all_date, columns=['date']).sort_values(by='date', ascending=True)
    all_date.index = all_date['date']
    new_factor = all_date.resample('M', closed='right', label='right').count()/(past+future+1)
    for i in range(1,13):
        month_average = np.nanmean([new_factor.loc[x] for x in new_factor.index if x.month ==i])
        for x in new_factor.index:
            if x.month == i:
                new_factor.loc[x] = new_factor.loc[x] - month_average
    all_date2.loc[[x for x in new_factor.index if x in all_date2.index]] = new_factor['date']

    return all_date2


if __name__ == '__main__':
    festival_date = pd.read_excel(r'x13\春节.xlsx', sheet_name='Sheet1')
    result = compute_factor(festival_date, 13,6, start_date='20150101', end_date='20200801')

