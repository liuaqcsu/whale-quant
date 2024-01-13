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


coins = coin.get_coin_list()
coins = list(coins.keys())
param = pd.read_csv('param.csv')
param_coins = param['coin']
coins.remove('BTC')
for param_coin in param_coins:
    coins.remove(param_coin)
    
for coin in coins[:45]:
    df = price.get_historical_data(coin, 'BTC', 'day', aggregate=1, limit=2000)
    df = pd.DataFrame(df)
    df = df[df['close']!=0]
    if len(df) > 150:
        df['Date'] = df['time'].apply(lambda x: x[:10])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['close','Date']]
        df.columns = ['y','ds']
        
        t1 = 0
        t2 = 0
        for change in [0.9,0.8,0.75,0.65,0.35]:
            days = 30
            t = [90,60,30]
            pred = pd.DataFrame()
            for te in t:
                data = df[['y','ds']][:-te]
                if te == 30:
                    true = df['y'][-30:]
                else:
                    true = df['y'][-te:-(te-days)]

                model = Prophet(changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=False,yearly_seasonality=False,
                                changepoint_range=change).fit(data)

                data_forecast = data['ds'] + pd.Timedelta(30, unit='days')
                data_forecast = pd.DataFrame(data_forecast)
                data_forecast.columns = ['ds']
                forecast = model.predict(data_forecast)
                forecast['yhat'] = np.where(forecast['yhat']<0,0.000001,forecast['yhat'])
                forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.000001,forecast['yhat_lower'])
                forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.000001,forecast['yhat_upper'])
                #model.plot(forecast);
                y = pd.DataFrame()
                y['True'] = true
                y['Forecasted'] = forecast['yhat'].iloc[-30:].values
                score = []
                for i in range(len(y)):
                    if y['True'].iloc[i] > y['Forecasted'].iloc[i]:
                        score.append(y['Forecasted'].iloc[i] / y['True'].iloc[i])
                    else:
                        score.append(y['True'].iloc[i] / y['Forecasted'].iloc[i])
                y['RMSE'] = np.sqrt(metrics.mean_squared_error(y['True'],y['Forecasted']))
                y['MAE'] = metrics.mean_absolute_error(y['True'],y['Forecasted'])
                y['Accuracy'] = score
                y['t'] = te
                pred = pd.concat([pred,y])

            res = pred[['RMSE','MAE','Accuracy']].groupby([pred['t']]).mean()
            res.index = pd.Series(['Last month','2nd last month','3rd last month']) + '-2019'
            res = pd.DataFrame(res)
            print(res)
            if t1 == 0 and t2 == 0:
                t1 = change
                t2 = res['Accuracy'].mean()

            else:
                if t2 < res['Accuracy'].mean():
                    t2 = res['Accuracy'].mean()
                    t1 = change
                    
        with open("param.csv", "a") as myfile:
            myfile.write('\n'+coin+',0.7,25,True,'+str(t1))