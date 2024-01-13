import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
from cryptocompy import coin,price
import tqdm
import seaborn as sns
sns.set();

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

import datetime
from datetime import datetime as dt
from datetime import timedelta

from math import sin, cos

from gaft import GAEngine
from gaft.components import BinaryIndividual,DecimalIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection, LinearRankingSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
from gaft.analysis import ConsoleOutput

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder


'''Creating Bins'''

def fun(x):
    if x < 50:
        return 'Strong Sell'
    elif x >= 50  and x < 57:
        return 'Sell'
    elif x >= 57 and x < 63:
        return 'Neutral'
    elif x >= 63 and x < 70:
        return 'Buy'
    else:
        return 'Strong Buy'
    
def fun1(x,v1,v2,v3,v4):
    if x < v1:
        return 'Strong Sell'
    elif x > v1 and x < v2:
        return 'Sell'
    elif x > v2 and x < v3:
        return 'Neutral'
    elif x > v3 and x < v4:
        return 'Buy'
    else:
        return 'Strong Buy'
    
    
def acc(x):
    if x.name == 'Strong Sell' or x.name == 'Sell':
        return (x['Strong Sell'] + x['Sell']) / x['Total']
    elif x.name == 'Neutral':
        return (x['Neutral']) / x['Total']
    else:
        return (x['Strong Buy'] + x['Buy']) / x['Total']

df = pd.DataFrame()
d1 = pd.DataFrame()
d2 = pd.DataFrame()
d3 = pd.DataFrame()
publish = pd.DataFrame()
candlestick = pd.DataFrame()

cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
cred = cred.to_dict('records')[0]
conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
cursor = conn.cursor()

#cursor.execute("SELECT * FROM ianbalina.ico_price_daily_summaries;") 
#m = cursor.fetchone()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])


query = '''SELECT * FROM ico_technical_analyses;'''
data = pd.read_sql_query(query, conn)
d1 = pd.concat([d1,data])

query = '''SELECT * FROM technical_analysis_scoring;'''
data = pd.read_sql_query(query, conn)
d2 = pd.concat([d2,data])

query = '''SELECT * FROM technical_analysis_weights;'''
data = pd.read_sql_query(query, conn)
d3 = pd.concat([d3,data])

query = '''SELECT * FROM ico_candlestick_patterns;'''
data = pd.read_sql_query(query, conn)
candlestick = pd.concat([candlestick,data])

query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
publish = pd.concat([publish,data])

conn.close() 

dates = d1['date'].unique()
d3['indicator'].iloc[11] = 'ichimoku'
candlestick['signal'] = candlestick['signal'].replace(['bullish','bearish','bearish_bullish'],[75,25,50])
d1 = d1[d1['date'] == dates[-31]]
print(dates[-31])

funda = publish['id'][publish['fundamentals_published_at'].notnull()].unique().tolist()
techn = publish['id'][publish['technology_published_at'].notnull()].unique().tolist()
coins = np.intersect1d(funda,techn)

t2 = pd.DataFrame()
for coin in coins:
    t1 = df[df['ico_id'] == coin]
    t1 = t1[t1['date'] >= dates[-31]][t1['date'] <= dates[-1]]
    t1 = t1.drop_duplicates(subset=['date'])
    t1 = t1.sort_values('date')
    if len(t1) > 1:
        x1 = t1['close'].iloc[-1] - t1['close'].iloc[0]
        x2 = (t1['close'].iloc[-1] - t1['close'].iloc[0]) / t1['close'].iloc[0] * 100
        t2 = pd.concat([t2,pd.DataFrame([coin,x1,x2])], axis=1)

t2 = t2.T
t2.columns = ['Coins','Price Change', '% Change']

d1['indicator'] = d1['indicator'].str.split('_').str.get(0)
d3 = d3.iloc[:-1]

def get_result(x = [4,2,1,11,1,3,7,2,1,10,7,21,25,6,0]):
    x = list(x)
    candlestick['weight'] = x[-1]
    print(x,len(x),len(d3))
    d3['weight'] = x[:-1]
    d4 = pd.merge(d1,d3[['indicator','weight']], on=['indicator'] ,how='left')
    d4['signal'] = d4['signal'].replace(['neutral','downward','upward','buy','sell','downgrade','bullish','bearish','bearish_bullish'],[50,25,75,100,0,25,75,25,50])


    temp = pd.DataFrame()
    for coin in coins:#df['ico_symbol'].unique().tolist():
        num = [coin]#df['ico_id'][df['ico_symbol'] == coin].unique().tolist()
        d5 = d4[d4['ico_id'].isin(num)]
        d5 = d5[['indicator','weight','signal']]

        t1 = candlestick[candlestick['ico_id'].isin(num)]
        t1 = t1[['pattern','weight','signal']]
        t1.columns = d5.columns
        t1 = t1.drop_duplicates('indicator')

        d5 = pd.concat([d5,t1])
        d5 = d5.fillna(0)
        d5['weight'].loc[d5['indicator'] == 'ichimoku'] = d5['weight'].loc[d5['indicator'] == 'ichimoku'] / 4
        temp = pd.concat([temp,pd.DataFrame([coin,sum(d5['weight'] * d5['signal'].astype('float64')) / d5['weight'].sum()])], axis=1)

    temp = temp.T
    temp.columns = ['Coins','Technical Score']
    temp.dropna(inplace=True)

    
    final = pd.merge(temp,t2)
    
    final['TA result'] = final['Technical Score'].apply(fun)
    final['Actual result'] = final['% Change'].apply(fun1,args=[-15,-5,5,15])
    final = final[final['Technical Score'] != 0]
    data = pd.crosstab(final['TA result'], final['Actual result'])

    data = data.reindex(['Strong Sell','Sell','Neutral','Buy','Strong Buy'])
    data = data.reindex(columns=['Strong Sell','Sell','Neutral','Buy','Strong Buy'])

    x1 = data.sum(axis=1)
    data['Total'] = x1
    x2 = data.T.apply(acc) * 100
    data['Result'] = (x1*x2)/100
    data['% Accurate'] = x2
    data.fillna(0,inplace=True)
    
    data.index = ['(-15)%','(-15 - -5)%', '(-5 - 5)%', '(5 - 15)%', '(15+)%']
    data.columns = ['(-15)%','(-15 - -5)%', '(-5 - 5)%', '(5 - 15)%', '(15+)%', 'Total', 'Result', '% Accurate']

    return data,final


# Genetic Algorithm
# Cost function -> get_result
# Fitness -> Maximize



indv_template = DecimalIndividual(ranges=[(0,25)]*15, eps=[1]*15)
population = Population(indv_template=indv_template, size=100).init()

selection = LinearRankingSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

@engine.fitness_register
#@engine.minimize
def fitness(indv):
    w = list(indv.solution)
    res,_ = get_result(w)
    res = (res['Result'].sum() - res['Result'].iloc[2]) / (res['Total'].sum() - res['Total'].iloc[2])
    res = float(abs(res))
    #print(w,': ',res)
    return res

def get_test_results(w1,w2,w3,w4):
    predictions = m1*w1 + m2*w2 + m3*w3 + m4*w4
    print('\nRMSE error: ', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    for i,prediction in enumerate(predictions[:15]):
        print ('Target: %s, Predicted: %s' % (Y_test.tolist()[i], abs(prediction)))

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}, best_individual: {}'.format(g, engine.ori_fmax, best_indv.solution)
        self.logger.info(msg)	

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)
    best_indv = engine.population.best_indv(engine.fitness)
    w = best_indv.solution
    res,final = get_result(w)
    pd.DataFrame([w]).to_csv('/home/tokenmetrics/data/technical/weights.csv')
    #res = res.drop('(-5 - 5)%')    
    res.to_csv('/home/tokenmetrics/data/technical/new.csv')
    old,_ = get_result([4,2,1,11,1,3,7,2,1,10,7,21,25,6,0])
    #old = old.drop('(-5 - 5)%')   
    old.to_csv('/home/tokenmetrics/data/technical/old.csv')
    final[['Coins','Technical Score']].to_csv('/home/tokenmetrics/data/technical/final_grade.csv', index=False)