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


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool, cv
import catboost
from sklearn.ensemble import RandomForestRegressor


#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import GridSearchCV
from numpy.random import seed
seed(1)


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

#import quantstats as qs

# extend pandas functionality with metrics, etc.
#qs.extend_pandas()

cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
cred = cred.to_dict('records')[0]

df = pd.DataFrame()
quant = pd.DataFrame()
publish = pd.DataFrame()

conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
cursor = conn.cursor()

query = '''SELECT * FROM quantstats_performance_data;'''
data = pd.read_sql_query(query, conn)
quant = pd.concat([quant,data])

query = '''SELECT * FROM ico_price_daily_summaries where currency='USD';'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])
    
query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
publish = pd.concat([publish,data])

conn.close()


rois = pd.DataFrame()
for ico_id in df['ico_id'].unique().tolist():
    m = df['close'][df['ico_id'] == ico_id]
    if len(m) > 30:
        initial_price = publish['ico_price'][publish['id'] == ico_id].iloc[0]
        if pd.isna(initial_price):
            initial_price = 0.01

        if initial_price == 0:
            m = 0
        else:
            m = (m.iloc[-1] - m.iloc[-30]) / m.iloc[-30] * 100
            #m = (m.iloc[-1] - initial_price) / initial_price * 100
    else:
        m = np.NaN
    rois = pd.concat([rois, pd.DataFrame([ico_id,m]).T]) 
       
rois.columns = ['ico_id','roi']
rois = rois.dropna()


quant = quant[quant['type'] == 'token']
quant = quant.drop(columns=['id','start_period','end_period','type','created_at', 'updated_at'])
quant = quant.fillna(0)

quant['beta'] = quant['beta'].astype('float')
quant['alpha'] = quant['alpha'].astype('float')

temp = pd.DataFrame()
temp['max'] = quant.max()
temp['min'] = quant.min()


def minmax_scaling(array, columns, min_val=0, max_val=1):
    """
    x - min / max - min
    
    array : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].
    columns : array-like, shape = [n_columns]
        Array-like with column names, e.g., ['col1', 'col2', ...]
        or column indices [0, 2, 4, ...]
    min_val : `int` or `float`, optional (default=`0`)
        minimum value after rescaling.
    max_val : `int` or `float`, optional (default=`1`)
        maximum value after rescaling.
    """
    ary_new = array.astype(float)
    if len(ary_new.shape) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
    else:
        raise AttributeError('Input array must be a pandas'
                             'DataFrame or NumPy array')

    numerator = ary_newt[:, columns] - ary_newt[:, columns].min(axis=0)
    denominator = (ary_newt[:, columns].max(axis=0) -
                   ary_newt[:, columns].min(axis=0))
    ary_newt[:, columns] = numerator / denominator

    if not min_val == 0 and not max_val == 1:
        ary_newt[:, columns] = (ary_newt[:, columns] *
                                (max_val - min_val) + min_val)

    return ary_newt[:, columns]


quant.iloc[:,1:] = minmax_scaling(quant.iloc[:,1:], quant.columns.tolist()[1:])
temp['norm_max'] = quant.max()
temp['norm_min'] = quant.min()


def get_bins(v):
    if v < 50:
        return 0
    elif v >= 50 and v < 60:
        return 1
    elif v >= 60 and v < 80:
        return 2
    elif v >= 80:
        return 3


def fun(weight,quan):
    z = quan.iloc[:,1:] * weight
    z['Score'] = z.sum(axis=1)
    z['Score'] = z['Score'] / z['Score'].max() * 100
    
    res = pd.DataFrame()
    res['token_id'] = quan['token_id']
    res['score'] = z['Score']
    rois.columns = ['token_id', 'roi']
    
    res = pd.merge(res, rois)
    res['roi_bins'] = pd.cut(res['roi'],[np.floor(res['roi'].min()),0,np.ceil(res['roi'].max())])
    
    res['score_bins'] = res['score'].apply(get_bins)
    res = res['score_bins'].groupby([res['score_bins'],res['roi_bins']]).count().unstack()
    
    x0 = []
    x1 = []
    for j in range(4):
        try:
            x0.append(res.iloc[j,0])
        except:
            x0.append(0)

    for j in range(4):
        try:
            x1.append(res.iloc[j,1])
        except:
            x1.append(0)

    x = pd.DataFrame()
    x[0] = x0
    x[1] = x1
    
    x.columns = res.columns
    res = x.copy(deep=True)
    
    y = res.sum(axis=1)
    y1 = (res.iloc[0:1,0] / y[:1]) 
    y2 = (res.iloc[1:,1] / y[1:])   
    
    res.index = ['F','C','B','A']
    res.columns = [-100,10000]
    
    res['Accuracy'] = y1.append(y2).fillna(0).astype('float64').values * 100
    accuracy = res.iloc[:1,0].append(res.iloc[1:,1]).sum() / res.iloc[:,:2].sum(axis=1).sum()
    
    return res, accuracy, z['Score']


indv_template = DecimalIndividual(ranges=[(0,1)]*54, eps=[0.01]*54)
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
    _, res, __ = fun(w, quant.copy(deep=True))
    res = float(abs(res))
    return res

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
    met, acc, z = fun(w, quant.copy(deep=True))

    wei = pd.DataFrame([w,quant.columns[1:]])
    wei = wei.T
    wei.columns = ['weights', 'features']
    wei.to_csv('/home/tokenmetrics/data/quant_grade/weights.csv', index=False)

    temp = pd.DataFrame([quant['token_id'], z])
    temp = temp.T
    temp = temp.sort_values('Score')
    temp['Score'][temp['Score']> 99] = 99
    temp.to_csv('/home/tokenmetrics/data/quant_grade/new_grade.csv', index=False)