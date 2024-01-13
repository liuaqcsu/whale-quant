import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
from cryptocompy import coin,price
import tqdm
import seaborn as sns
sns.set();

import io,os
import gc
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

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_AFFINITY"] = "disabled"

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import pandas as pd

df_coin = pd.DataFrame()
mkt = pd.DataFrame()
ico_prices = pd.DataFrame()

conn = MySQLdb.connect(host='tokenmetrics.cxuzrhvtziar.us-east-1.rds.amazonaws.com', user='admin', passwd='WiG8Rled2cTvZ5JibJui',db='tokenmetrics')

cursor = conn.cursor()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df_coin = pd.concat([df_coin,data])

query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
mkt = pd.concat([mkt,data])

query = '''SELECT * FROM ico_prices;'''
data = pd.read_sql_query(query, conn)
ico_prices = pd.concat([ico_prices,data])

conn.close() 

#coins = df_coin['close'].groupby([df_coin['ico_symbol']]).mean().sort_values(ascending=False)[37:65]
#coins = df_coin['ico_symbol'].unique().tolist()[:105]
#coins = coins.index.tolist()

#files = glob.glob('models/*_res.csv')
#files = pd.Series(files).str.split('/').str.get(1)
#coins =  pd.Series(files).str.split('_').str.get(0).tolist()

#coins = coin.get_coin_list()
#coins = list(coins.keys())
#param = pd.read_csv('param.csv')
#param_coins = param['coin']
#coins.remove('BTC')
#for param_coin in param_coins:
#    coins.remove(param_coin)

#df_coin['ico_symbol'].unique().tolist()

mkt = mkt.sort_values('ico_market_cap',ascending=False)
#coins = mkt['symbol'][:250].tolist()
#coins = list(filter(None,coins))
coins = df_coin['ico_symbol'].unique().tolist()
print(coins)
ico_prices = ico_prices[ico_prices['currency'] == 'USD']

for coin in coins:
    try:
        print(coin)
        predict_data = pd.DataFrame()

        #df = price.get_historical_data(coin, 'BTC', 'day', aggregate=1, limit=2000)
        df = df_coin[df_coin['ico_symbol'] == coin][df_coin['currency'] == 'USD']
        df = df.drop_duplicates(subset='date').sort_values(by='date')

        ico_ids = df_coin['ico_id'][df_coin['ico_symbol'] == coin].unique().tolist()
        h = ico_prices[ico_prices['ico_id'].isin(ico_ids)]
        h = h.sort_values('updated_at', ascending=False)

        if len(df) > 0:
            df = df[df['close']!=0]
        if len(df) > 90:
            #df['Date'] = df['time'].apply(lambda x: x[:10])
            df['date'] = pd.to_datetime(df['date'])
            df = df[['close','date']]
            df.columns = ['y','ds']
            df = df.sort_values(by='ds')
            df = df[-370:]
            t1 = 0
            t2 = 0
            for change in [0.9,0.75,0.65,0.5,0.35,0.2]:
                if len(df) > 360:
                    t = [i for i in range(30,30*12+1,30)]
                elif len(df)>= 30:
                    t = int(len(df)/30)
                    t = [i for i in range(30,30*t+1,30)]

                days = 30
                #t = [30,20,10]
                pred = pd.DataFrame()
                for te in t:
                    data = df[['y','ds']][:-te]
                    if te == 30:
                        true = df['y'][-30:]
                    else:
                        true = df['y'][-te:-(te-days)]

                    model = Prophet(changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=False,yearly_seasonality=False,
                                    changepoint_range=change).fit(data)

                    #data_forecast = data['ds'] + pd.Timedelta(30, unit='days')
                    y = pd.date_range(start=data['ds'].iloc[-1],periods=31)
                    data_forecast = pd.concat([data['ds'],pd.Series(y)[1:]])

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
                    y['date'] = forecast['ds'].iloc[-30:].values
                    print(y)
                    score = []
                    for i in range(len(y)):
                        if y['True'].iloc[i] > y['Forecasted'].iloc[i]:
                            score.append(y['Forecasted'].iloc[i] / y['True'].iloc[i])
                        else:
                            score.append(y['True'].iloc[i] / y['Forecasted'].iloc[i])
                    y['RMSE'] = np.sqrt(metrics.mean_squared_error(y['True'],y['Forecasted']))
                    y['MAE'] = metrics.mean_absolute_error(y['True'],y['Forecasted'])
                    y['Accuracy'] =score
                    y['t'] = te
                    pred = pd.concat([pred,y])

                res = pred[['RMSE','MAE','Accuracy']].groupby([pred['t']]).mean()
                #res.index = pd.Series(['Last month','2nd last month','3rd last month']) + '-2019'
                res = pd.DataFrame(res)
                print(res)

                predict_data = pd.concat([predict_data, pred])
                new_predict = pd.DataFrame()
                for t in predict_data['t'].unique().tolist():
                    x = predict_data[predict_data['t'] == t].sort_values('MAE').iloc[:30]
                    new_predict = pd.concat([new_predict,x])

                if t1 == 0 and t2 == 0:
                    t1 = change
                    t2 = res['MAE'].iloc[0] #res['Accuracy'].mean()

                else:
                    if t2 > res['MAE'].iloc[0]: #res['Accuracy'].mean():
                        t2 = res['MAE'].iloc[0] #res['Accuracy'].mean()
                        t1 = change

            #with open("param.csv", "a") as myfile:
            #    myfile.write('\n'+coin+',0.7,25,True,'+str(t1))

            if coin == 'BTC' or coin == 'ETH':
                t1 = 0.20

            days = 10
            #t = [30,20,10]
            if len(df) > 360:
                t = [i for i in range(10,10*12+1,10)]
            elif len(df)>= 30:
                t = int(len(df)/30)
                t = [i for i in range(10,10*t+1,10)]
            pred = pd.DataFrame()
            for te in t:
                data = df[['y','ds']][:-te]
                if te == 10:
                    true = df['y'][-10:]
                else:
                    true = df['y'][-te:-(te-days)]

                model = Prophet(changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=False,yearly_seasonality=False,
                                    changepoint_range=t1).fit(data)

                #data_forecast = data['ds'] + pd.Timedelta(30, unit='days')
                y = pd.date_range(start=data['ds'].iloc[-1],periods=10)
                data_forecast = pd.concat([data['ds'],pd.Series(y)[1:]])


                data_forecast = pd.DataFrame(data_forecast)
                data_forecast.columns = ['ds']
                forecast = model.predict(data_forecast)
                forecast['yhat'] = np.where(forecast['yhat']<0,0.000001,forecast['yhat'])
                forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.000001,forecast['yhat_lower'])
                forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.000001,forecast['yhat_upper'])
                #model.plot(forecast);
                y = pd.DataFrame()
                y['True'] = true
                y['Forecasted'] = forecast['yhat'].iloc[-10:].values
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

            model = Prophet(changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=False,yearly_seasonality=False,
                                    changepoint_range=t1).fit(df)
            #data_forecast = df['ds'] + pd.Timedelta(30, unit='days')
            y = pd.date_range(start=df['ds'].iloc[-1],periods=30)
            data_forecast = pd.concat([df['ds'],pd.Series(y)[1:]])

            #print(df['ds'])
            #print(data_forecast) 
            data_forecast = pd.DataFrame(data_forecast)
            data_forecast.columns = ['ds']
            forecast = model.predict(data_forecast) 
            forecast['yhat'] = np.where(forecast['yhat']<0,0.000001,forecast['yhat'])
            forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.000001,forecast['yhat_lower'])
            forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.000001,forecast['yhat_upper'])
            #model.plot(forecast); 

            res = pred[['RMSE','MAE','Accuracy']].groupby([pred['t']]).mean()
            #res.index = pd.Series(['Last month','2nd last month','3rd last month']) + '-2019'
            #res.index = pd.Series([str(df['ds'].iloc[-30])[:10]+'-'+str(df['ds'].iloc[-1])[:10],  str(df['ds'].iloc[-60])[:10]+'-'+str(df['ds'].iloc[-30])[:10], str(df['ds'].iloc[-90])[:10]+'-'+str(df['ds'].iloc[-60])[:10]])

            if len(res)<=1:
               ind = [str(df['ds'].iloc[-30])[:10]+'-'+str(df['ds'].iloc[-1])[:10]]
            else:
               ind = [str(df['ds'].iloc[-30])[:10]+'-'+str(df['ds'].iloc[-1])[:10]]
            for i in range(2,len(res)+1):
               ind.append(str(df['ds'].iloc[-30*i])[:10]+'-'+str(df['ds'].iloc[-30*(i-1)])[:10])
            res = res.iloc[::-1]
            res.index = pd.Series(ind)
            res = pd.DataFrame(res)

            forecast = forecast.sort_values('ds')
            final = forecast[['ds']].iloc[-30:]
            final['weekday'] = final['ds'].dt.weekday
            final['weekday'] = final['weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})
            final[['Price','Low','High']] = forecast[['yhat','yhat_lower','yhat_upper']].iloc[-30:]
            final.set_index('ds',inplace=True)

            res.to_csv('/home/tokenmetrics/data/models/'+coin+'_res.csv')
            final['Price'].iloc[0] = h['price'].iloc[0]
            final.to_csv('/home/tokenmetrics/data/models/'+coin+'_final.csv')
            new_predict.to_csv('/home/tokenmetrics/data/models/'+coin+'_predict.csv')
 
    except:
        pass
    gc.enable()
    gc.collect()