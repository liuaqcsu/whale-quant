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


from tqdm import tqdm_notebook
from livelossplot.keras import PlotLossesCallback

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

import quantstats as qs

# extend pandas functionality with metrics, etc.
qs.extend_pandas()


df = pd.DataFrame()
d1 = pd.DataFrame()
d2 = pd.DataFrame()
publish = pd.DataFrame()

cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
cred = cred.to_dict('records')[0]

conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
cursor = conn.cursor()
    
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])

query = '''SELECT * FROM ico_grade_history;'''
data = pd.read_sql_query(query, conn)
d1 = pd.concat([d1,data])

query = '''SELECT * FROM ico_analytics_history;'''
data = pd.read_sql_query(query, conn)
d2 = pd.concat([d2,data])

query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
publish = pd.concat([publish,data])

conn.close() 

df = df[df['currency'] == 'USD']
df.set_index('date',inplace=True)

d1 = d1[d1['date'] == d1['date'].unique()[-1]]
d2 = d2[d2['analytics_date'] == d2['analytics_date'].unique()[-1]]

# Calculating alphas for every coin
alphas = pd.DataFrame()
for ico_id in df['ico_id'].unique().tolist():
    if ico_id in [   1, 3375, 3431, 3680]:
        alphas = pd.concat([alphas, pd.DataFrame([ico_id, 0]).T])
    else:
        bench = df['close'][df['ico_id'] == 3375].drop_duplicates().pct_change().fillna(0)
        returns = df['close'][df['ico_id'] == ico_id].drop_duplicates().pct_change().fillna(0)
        bench.sort_index(inplace=True)
        returns.sort_index(inplace=True)
        #returns = returns.iloc[-30:]
        #bench = bench[bench.index.isin(returns.index)]
        #k = qs.stats.greeks(returns,bench, periods=30)['alpha']

        beta = np.correlate(bench, returns)[0] * (returns.std() / bench.std())
        k = returns.mean() - beta * bench.mean() 

        alphas = pd.concat([alphas, pd.DataFrame([ico_id, k*252]).T])

# Calculating all time ROI for every coin
rois = pd.DataFrame()
for ico_id in df['ico_id'].unique().tolist():
    m = df['close'][df['ico_id'] == ico_id]
    initial_price = publish['ico_price'][publish['id'] == ico_id].iloc[0]
    if pd.isna(initial_price):
        initial_price = 0.01

    if initial_price == 0:
        m = 0
    else:
        m = (m.iloc[-1] - initial_price) / initial_price * 100
    rois = pd.concat([rois, pd.DataFrame([ico_id,m]).T]) 
       
rois.columns = ['ico_id','roi']

f = pd.read_csv('/home/tokenmetrics/data/fundamental/new_grade_roi.csv')
t = pd.read_csv('/home/tokenmetrics/data/technology/new_grade_roi.csv')
te = pd.read_csv('/home/tokenmetrics/data/technical/final_grade.csv')
f = f[['Coin','Score']]
f.columns = ['ico_id','tokenmetrics_score']
t = t[['Coin','Score']]
t.columns = ['ico_id','tech_scorecard_score']
te.columns = ['ico_id','trading_score']

d3 = pd.merge(f,df[['ico_id','ico_symbol']],on='ico_id',how='inner').drop_duplicates()
d3 = pd.merge(d3,t, on='ico_id',how='inner').drop_duplicates()
d3 = pd.merge(d3,te, on='ico_id',how='inner').drop_duplicates()

alphas = rois.copy(deep=True)
alphas.columns = ['ico_id','alpha']
d3 = pd.merge(d3,alphas, on='ico_id',how='inner').drop_duplicates()


#d3 = pd.merge(d1[['ico_id','grade','tokenmetrics_score','trading_score','tech_scorecard_score']],df[['ico_id','ico_symbol']],
#              on='ico_id',how='inner')

#d3 = d3.drop_duplicates()
#alpha = []
#for coin in d3['ico_id'].tolist():
#    temp = d2['alpha'][d2['ico_id'] == coin].values
#    if len(temp) == 0:
#        alpha.append(np.NaN)
#    else:
#        alpha.append(temp[0])

#d3['alpha'] = alpha

d3.dropna(inplace=True)

# Use grade only for coins which are published
funda = publish['id'][publish['fundamentals_published_at'].notnull()].unique().tolist()
techn = publish['id'][publish['technology_published_at'].notnull()].unique().tolist()
coins = np.intersect1d(funda,techn)
d3 = d3[d3['ico_id'].isin(coins)]

def get_bins(v):
    if v < 60:
        return 5
    elif v >= 60 and v < 70:
        return 6
    elif v >= 70 and v < 80:
        return 7
    elif v >= 80 and v < 90:
        return 8
    elif v > 90:
        return 9

def fun(w1,w2,d3):
    '''Passing weights and calculating grades and accuracy.
       For stable coins use all 3 grades and 
       For non stable coins use fundamental and technology grade only.'''
    w3 = 100-w1-w2
    #print(w1,w2,w3)

    stable_coin = publish['id'][publish['is_stablecoin'] == 11]  #[3379,3427,3396,3457,3401,3421]
    d4 = d3[~d3['ico_id'].isin(stable_coin)]
    d5 = d3[d3['ico_id'].isin(stable_coin)]
    d4['grade'] = (d4['tokenmetrics_score'] * w1 + d4['tech_scorecard_score']*w2 + d4['trading_score'] * w3) / 100
    d5['grade'] = (d5['tokenmetrics_score'] * w1 + d5['tech_scorecard_score']*w2 ) / (w1+w2)
    d3 = pd.concat([d4,d5])

    #d3['grade'] = (d3['tokenmetrics_score'] * w1 + d3['tech_scorecard_score']*w2 + d3['trading_score'] * w3) / 100
    
    d3['Score_bins'] = d3['grade'].apply(get_bins)
    d3['Alpha_bins'] = pd.cut(d3['alpha'],[np.floor(d3['alpha'].min()),0,np.ceil(d3['alpha'].max())])
    
    final = d3['Alpha_bins'].groupby([d3['Score_bins'],d3['Alpha_bins']]).count().unstack()
    #print(final)
    x0 = []
    x1 = []
    for j in range(5):
        try:
            x0.append(final.iloc[j,0])
        except:
            x0.append(0)

    for j in range(5):
        try:
            x1.append(final.iloc[j,1])
        except:
            x1.append(0)

    x = pd.DataFrame()
    x[0] = x0
    x[1] = x1
    #print(x)
    #x.index = ['(0,10]','(10,20]','(20,30]','(30,40]','(40,50]','(50,60]','(60,70]','(70,80]','(80,90]','(90,100]']
    x.index = ['F','D','C','B','A']
    x.fillna(0,inplace=True)

    y = x.sum(axis=1)
    y1 = (x.iloc[0:1,0] / y[:1]) 
    y2 = (x.iloc[1:,1] / y[1:])
    
    x['Accuracy'] = y1.append(y2).fillna(0).astype('float64').values * 100
    x.columns = [str(final.columns[0]),str(final.columns[1]),'Accuracy']
    
    accuracy = x.iloc[:1,0].append(x.iloc[1:,1]).sum() / x.iloc[:,:2].sum(axis=1).sum()
    return x, accuracy, d3#[['ico_id','grade']]


indv_template = DecimalIndividual(ranges=[(1,50)]*2, eps=[1]*2)
population = Population(indv_template=indv_template, size=500).init()

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
    _, res,x = fun(w[0], w[1], d3)
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
    res,_,new_grade = fun(w[0], w[1], d3)
    x,_,old_grade = fun(33,33,d3)
    res.to_csv('/home/tokenmetrics/data/final_grade/New.csv')
    x.to_csv('/home/tokenmetrics/data/final_grade/Old.csv')
    pd.DataFrame([['Fundamental','Technology','Technical'],[w[0],w[1],100-w[0]-w[1]]]).to_csv('/home/tokenmetrics/data/final_grade/weights.csv') 
    new_grade[['ico_id','grade']].to_csv('/home/tokenmetrics/data/final_grade/new_grade.csv')

    y = new_grade['alpha'].groupby([new_grade['Score_bins']]).mean()
    y = pd.DataFrame(y)
    x4 = pd.merge(new_grade,rois, left_on='ico_id', right_on='ico_id', how='inner')
    x4 = x4.drop_duplicates('ico_id')
    y['ROI'] = x4['roi'].groupby([x4['Score_bins']]).mean()
    y = y.rename(index={5:'F',6:'D',7:'C',8:'B',9:'A'})
    y.to_csv('/home/tokenmetrics/data/final_grade/alpha_roi.csv')