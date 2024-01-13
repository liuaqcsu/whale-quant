import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
from cryptocompy import coin,price
import tqdm

import io
import requests
import json
from datetime import datetime
import time
from time import gmtime, strftime

from sklearn import metrics
from fbprophet import Prophet
plt.rcParams["figure.figsize"] = [14,8]
pd.set_option('float_format', '{:f}'.format)

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

df = pd.read_csv('history.csv')
df['time'] = df['time'].apply(lambda x: x[:10])
df.set_index('time',inplace=True)


coins = df['coin'].unique().tolist()  # + ['XYZ']

final = pd.DataFrame()

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")

for c in coins:
    temp = df[df['coin'] == c]
    
    if len(temp) != 0:
        date_format = "%Y-%m-%d"
        a = datetime.strptime(temp.index[-1], date_format)
        b = datetime.strptime(dt_string, date_format)
        delta = b - a

        d1 = price.get_historical_data(c, 'USD', 'day', aggregate=1, limit=delta.days)
        d1 = pd.DataFrame(d1)
        if len(d1) > 0:
            d1['time'] = d1['time'].apply(lambda x: x[:10])
            d1.set_index('time',inplace=True)

            d1['coin'] = c

        d1 = pd.concat([temp,d1])
        d1 = d1[~d1.index.duplicated(keep='first')]
    else:
        d1 = price.get_historical_data(c, 'USD', 'day', aggregate=1, limit=2000)
        d1 = pd.DataFrame(d1)
        if len(d1) > 0:
            d1 = d1[d1['close']!=0]
            d1['time'] = d1['time'].apply(lambda x: x[:10])
            d1.set_index('time',inplace=True)
            d1['coin'] = c
    
    final = pd.concat([final,d1])
    
final.to_csv('history.csv')

benchmark = ['BTC']
ratio = [1]

final.index = pd.to_datetime(final.index)

for c in final['coin'].unique().tolist():
    toke = final[final['coin'] == c]
    toke = toke['close'][toke['close'] != 0]
    toke = toke.pct_change().fillna(0)

    tmp = pd.DataFrame()
    for i in range(len(benchmark)):
        temp = final[final['coin'] == benchmark[i]]
        temp = temp['close'][temp['close'] != 0]
        temp = temp.pct_change().fillna(0) * ratio[i]
        tmp = pd.concat([tmp,pd.DataFrame(temp)], axis=1)
        tmp = tmp.sum(axis=1)

    c2 = "+".join(benchmark)
    qs.reports.html(toke,tmp,file='reports/'+c+'.html',c1=c,c2=c2)
    
    