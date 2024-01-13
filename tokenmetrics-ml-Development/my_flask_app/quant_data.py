import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
from cryptocompy import coin,price
import tqdm
import os, shutil
import gc
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

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder

df = pd.DataFrame()
icos = pd.DataFrame()

#with SSHTunnelForwarder(('206.189.186.74', 22), ssh_password='crypto1234', ssh_username='aagam', remote_bind_address=('127.0.0.1', 3306)) as server:
# Reading Data

cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
cred = cred.to_dict('records')[0]

conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
cursor = conn.cursor()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])

query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
icos = pd.concat([icos,data])
conn.close() 

df = df[df['currency'] == 'USD']

icos = icos[['id','name']]
icos = pd.merge(icos,df[['ico_id','ico_symbol']], left_on = 'id', right_on = 'ico_id')

final = df.drop_duplicates(subset=['ico_symbol','date']).sort_values(by='date')

benchmark = ['BTC']  # Benchmark Bitcoin
ratio = [1]    # Multiple ratios if different coins as group compare together against BTC

final.index = pd.to_datetime(final['date'])
x_coins = final['ico_symbol'].unique().tolist()

'''Creating quanstats reports for each coins'''
for c in x_coins:
    try:
        print(c)
        title = icos['name'][icos['ico_symbol'] == c].unique()
        if len(title) < 1:
            title = ''
        else:
            title = title[0]
        title = title + ' (' + c + ')' + ' Performance Metrics'

        toke = final[final['ico_symbol'] == c]
        toke = toke['close'][toke['close'] != 0]
        toke = toke.pct_change().fillna(0)

        tmp = pd.DataFrame()
        for i in range(len(benchmark)):
            temp = final[final['ico_symbol'] == benchmark[i]]
            temp = temp['close'][temp['close'] != 0]
            temp = temp.pct_change().fillna(0) * ratio[i]
            tmp = pd.concat([tmp,pd.DataFrame(temp)], axis=1)
            tmp = tmp.sum(axis=1)

        c2 = "+".join(benchmark)
        qs.reports.html(toke,tmp,file='/home/tokenmetrics/data/quant/reports/'+c+'.html',c1=c,c2=c2, title = title)
    
        files = glob.glob('/home/tokenmetrics/data/quant/temp/'+'*.csv')
        if os.path.isdir('/home/tokenmetrics/data/quant/temp/'+c):
            shutil.rmtree('/home/tokenmetrics/data/quant/temp/'+c)
        os.mkdir('/home/tokenmetrics/data/quant/temp/'+c)        

        for file in files:
    	    shutil.move(file, '/home/tokenmetrics/data/quant/temp/'+c+'/'+file[35:])

        gc.enable()
        gc.collect()

    except:
        pass