import numpy as np
import pandas as pd

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder

df = pd.DataFrame()
#with SSHTunnelForwarder(('206.189.186.74', 22), ssh_password='crypto1234', ssh_username='aagam', remote_bind_address=('127.0.0.1', 3306)) as server:
    

'''Reading Data'''

cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
cred = cred.to_dict('records')[0]

conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
cursor = conn.cursor()
    
    #cursor.execute("SELECT * FROM ianbalina.ico_price_daily_summaries;") 
    #m = cursor.fetchone()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])
conn.close() 

df = df.drop_duplicates(subset=['ico_symbol','date']).sort_values(by='date')

temp = df['close'].groupby([df['date'],df['ico_symbol']]).mean()
temp = temp.unstack().sort_index()

#temp = temp[-90:]
#temp.dropna(inplace=True)
temp = temp.corr('pearson', min_periods=10)#.abs()
temp.to_csv('/home/tokenmetrics/data/corr.csv') # Saving correlation in a file



