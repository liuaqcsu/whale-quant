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


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool, cv
import catboost
from sklearn.ensemble import RandomForestRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import GridSearchCV
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU

from keras import optimizers
from keras import backend as K

adam = optimizers.Adam(lr=0.01, clipnorm=1.)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
def nn_r2_score(y_true, y_pred):
    total_error = K.sum(K.square( y_true - K.mean(y_true) ) )
    residual_error = K.sum(K.square( y_true - y_pred ))
    R_squared = 1 - (residual_error / total_error)
    return -R_squared


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
d3 = pd.DataFrame()
d4 = pd.DataFrame()
d5 = pd.DataFrame()

d6 = pd.DataFrame()

#with SSHTunnelForwarder(('206.189.186.74', 22), ssh_password='crypto1234', ssh_username='aagam', remote_bind_address=('127.0.0.1', 3306)) as server:

conn = MySQLdb.connect(host='tokenmetrics.cluster-cxuzrhvtziar.us-east-1.rds.amazonaws.com', user='admin', passwd='WiG8Rled2cTvZ5JibJui',db='tokenmetrics')
cursor = conn.cursor()
    
    #cursor.execute("SELECT * FROM ianbalina.ico_price_daily_summaries;") 
    #m = cursor.fetchone()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])
    
query = '''SELECT * FROM ico_grade_history;'''
data = pd.read_sql_query(query, conn)
d1 = pd.concat([d1,data])
    
query = '''SELECT * FROM ico_analytics;'''
data = pd.read_sql_query(query, conn)
d2 = pd.concat([d2,data])
    
query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
d3 = pd.concat([d3,data])
    
query = '''SELECT * FROM ico_analytics_history;'''
data = pd.read_sql_query(query, conn)
d4 = pd.concat([d4,data])
   
query = '''SELECT * FROM ico_grade_history_scores;'''
data = pd.read_sql_query(query, conn)
d5 = pd.concat([d5,data])
    
query = '''SELECT * FROM tech_scorecard_options;'''
data = pd.read_sql_query(query, conn)
d6 = pd.concat([d6,data])
    
conn.close() 

df = df[df['currency'] == 'USD']
df.set_index('date',inplace=True)
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

rois = pd.DataFrame()
for ico_id in df['ico_id'].unique().tolist():
    m = df['close'][df['ico_id'] == ico_id]
    initial_price = d3['ico_price'][d3['id'] == ico_id].iloc[0]
    m = (m.iloc[-1] - initial_price) / initial_price * 100
    rois = pd.concat([rois, pd.DataFrame([ico_id,m]).T]) 
       
rois.columns = ['ico_id','roi']

d4 = d4[d4['analytics_date'] == d4['analytics_date'].unique()[-1]]


def get_bins(v):
    if v < 60:
        return 5
    elif v >= 60 and v < 70:
        return 6
    elif v >= 70 and v < 80:
        return 7
    elif v >= 80 and v < 90:
        return 8
    elif v >= 90:
        return 9

def get_technology_accuracy(weights, max_score, tech):
    t1 = pd.DataFrame()

    for i in tech['ico_id'].unique().tolist():
        temp = tech[tech['ico_id'] == i]
        temp = temp.sort_values('question_number')
        temp = temp.drop_duplicates('name')
        score = temp['score'] * weights[:len(temp)] 
        
        t1 = pd.concat([t1,pd.DataFrame([i, score.sum() / max_score * 100 / sum(weights[:len(temp)]) * len(temp)]).T])
        
    t1.columns = ['Coin','Score']
    #ids = tech['ico_id'][tech['ico_symbol'].isin(t1['Coin'])].unique()
    alpha = []
    for coin in t1['Coin']:
        a = alphas[1][alphas[0] == coin]
        if len(a) == 0:
            alpha.append(0)
        else:
            alpha.append(a.iloc[0])   
    
    t1['Alpha'] = alpha

    t1['Score_bins'] = t1['Score'].apply(get_bins)
    t1['Alpha_bins'] = pd.cut(t1['Alpha'],[np.floor(t1['Alpha'].min()),0,np.ceil(t1['Alpha'].max())])

    final = t1['Alpha_bins'].groupby([t1['Score_bins'],t1['Alpha_bins']]).count().unstack()

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

    #accuracy = x.iloc[:1,0].append(x.iloc[1:,1]).sum() / x.iloc[:,:2].sum(axis=1).sum()
    accuracy = x.iloc[1:,1].sum() / x.iloc[1:,:2].sum(axis=1).sum()
    return x, accuracy, t1


tech = d5[d5['type'] == 2]
d1 = d1[d1['date'] == d1['date'].sort_values().iloc[-1]]
tech = tech[tech['ico_grade_history_id'].isin(d1['id'].unique())]
tech = pd.merge(tech,d1[['id','ico_id']], left_on = 'ico_grade_history_id', right_on = 'id', how='left')
tech = tech.drop('id_y', axis=1)
tech.rename(columns={'id_x':'id'}, inplace=True)

ids = d3['id'][d3['technology_published_at'].notnull()].unique().tolist()
coin = []
for id in ids:
    k = df['ico_symbol'][df['ico_id'] == id].unique().tolist()
    if len(k) > 0:
        coin.append(k[0])
    
tech = tech[tech['ico_id'].isin(ids)]
temp = pd.DataFrame([ids,coin]).T
temp.columns = ['ico_id','ico_symbol']

tech = pd.merge(tech,temp)
tech['question_number'] = tech['name'].str.split('_').str.get(-1)

max_score = d6['score'].groupby([d6['tech_scorecard_question_id']]).max().sum()



# Genetic Algorithm
# Cost function -> get_technology_accuracy
# Fitness -> Maximize


indv_template = DecimalIndividual(ranges=[(0,10)]*28, eps=[1]*28)
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
    _, res, x = get_technology_accuracy(w, max_score, tech)
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
    res, _, new_grade = get_technology_accuracy(w, max_score, tech)
    x, _, old_grade = get_technology_accuracy(np.ones(28), max_score, tech)
    res.to_csv('/home/tokenmetrics/data/technology/New.csv')
    x.to_csv('/home/tokenmetrics/data/technology/Old.csv')
    pd.DataFrame([pd.Series(tech['question_number'].unique()).sort_values().tolist(), w]).to_csv('/home/tokenmetrics/data/technology/weights.csv') 
    new_grade[['Coin','Score']].to_csv('/home/tokenmetrics/data/technology/new_grade.csv')

    y = new_grade['Alpha'].groupby([new_grade['Score_bins']]).mean()
    y = pd.DataFrame(y)
    x4 = pd.merge(new_grade,rois, left_on='Coin', right_on='ico_id', how='inner')
    x4 = x4.drop_duplicates('Coin')
    y['ROI'] = x4['roi'].groupby([x4['Score_bins']]).mean()
    y = y.rename(index={5:'F',6:'D',7:'C',8:'B',9:'A'})
    y.to_csv('/home/tokenmetrics/data/technology/alpha_roi.csv')