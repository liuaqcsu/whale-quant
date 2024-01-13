import flask
from flask import request, jsonify, Response, render_template, session, redirect
import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import glob
import json
#import MySQLdb
import os
import warnings
warnings.filterwarnings("ignore")
import quantstats as qs
from cryptocompy import coin,price

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# source env/bin/activate
#  sudo fuser -k 8000/tcp   -> to kill past processes running on this port
#  gunicorn quanstats:app --daemon &

#http://localhost:5000/api/quantstats/?token=XRP&benchmark=[EOS,BTC]&ratio=[0.5,0.5]
#Api:   http://localhost:5000/api/quantstats/?token=BTC&benchmark=['ETH','BTC','XRP']&ratio=[0.7,0.1,0.2]


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

#df = pd.read_csv('history.csv')
#df['time'] = df['time'].apply(lambda x: x[:10])
#df.set_index('time',inplace=True)
#df.index = pd.to_datetime(df.index)

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/quantstats/', methods=['GET'])
def token():
    df = pd.read_csv('history.csv')
    df['time'] = df['time'].apply(lambda x: x[:10])
    df.set_index('time',inplace=True)
    df.index = pd.to_datetime(df.index)
    
    bar = request.args.to_dict()
    toke = bar['token']
    benchmark = bar['benchmark'][1:-1].split(',')
    ratio = list(map(float,bar['ratio'][1:-1].split(',')))
    print(toke,benchmark,ratio)
    status = True
    coin_not_available = ''
    for con in benchmark:
        print(con)
        if con not in df['coin'].unique().tolist():
            status = False
            coin_not_available = con
    if status == False:
        return coin_not_available + ' Token not available for Benchmark'
    elif toke not in df['coin'].unique().tolist():
        return 'Token not available'
    else:
        toke = df[df['coin'] == toke]
        toke = toke['close'][toke['close'] != 0]
        toke['returns'] = toke.pct_change().fillna(0)
        
        tmp = pd.DataFrame()
        for i in range(len(benchmark)):
            temp = df[df['coin'] == benchmark[i]]
            temp = temp['close'][temp['close'] != 0]
            temp = temp.pct_change().fillna(0) * ratio[i]
            tmp = pd.concat([tmp,pd.DataFrame(temp)], axis=1)
            tmp = tmp.sum(axis=1)
            
        '''
        for i in range(len(benchmark)):
            temp = df[df['coin'] == benchmark[i]]
            temp = temp['close'][temp['close'] != 0]
            if len(tmp) < 1:
                tmp = temp.pct_change().fillna(0) * ratio[i]
                tmp = tmp.fillna(0)
            else:
                tmp = tmp + temp.pct_change().fillna(0) * ratio[i]
                tmp = tmp.fillna(0)
        '''
        #result = result.to_dict()
        c2 = "+".join(benchmark)
        qs.reports.html(toke['returns'],tmp,file='reports.html',c1=bar['token'],c2=c2)
        content = get_file('reports.html')
        return Response(content, mimetype="text/html")
    
@app.route('/api/predictions/', methods=['GET'])
def token2():
    bar = request.args.to_dict()
    coin = bar['token']
    param = pd.read_csv('param.csv')
    if coin not in param['coin'].tolist():
        return 'Model for ' + coin + ' not available.' + 'Models working on '+ str(param['coin'].tolist())
    else:
        m1 = param['m1'][param['coin'] == coin].iloc[0]
        m2 = param['m2'][param['coin'] == coin].iloc[0]
        m3 = param['m3'][param['coin'] == coin].iloc[0]
        m4 = param['m4'][param['coin'] == coin].iloc[0]
        df = price.get_historical_data(coin, 'BTC', 'day', aggregate=1, limit=2000)
        df = pd.DataFrame(df)
        df = df[df['close']!=0]
        df['Date'] = df['time'].apply(lambda x: x[:10])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['close','Date']]
        df.columns = ['y','ds']


        days = 30
        t = [90,60,30]
        pred = pd.DataFrame()
        for te in t:
            data = df[['y','ds']][:-te]
            if te == 30:
                true = df['y'][-30:]
            else:
                true = df['y'][-te:-(te-days)]

            model = Prophet(changepoint_prior_scale=m1,n_changepoints=m2, daily_seasonality=m3,changepoint_range=m4).fit(data)

            data_forecast = data['ds'] + pd.Timedelta(30, unit='days')
            data_forecast = pd.DataFrame(data_forecast)
            data_forecast.columns = ['ds']
            forecast = model.predict(data_forecast)
            forecast['yhat'] = np.where(forecast['yhat']<0,0.00001,forecast['yhat'])
            forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.00001,forecast['yhat_lower'])
            forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.00001,forecast['yhat_upper'])
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

        model = Prophet(changepoint_prior_scale=m1,n_changepoints=m2, daily_seasonality=m3,changepoint_range=m4).fit(df)
        data_forecast = df['ds'] + pd.Timedelta(30, unit='days')
        data_forecast = pd.DataFrame(data_forecast)
        data_forecast.columns = ['ds']
        forecast = model.predict(data_forecast)
        forecast['yhat'] = np.where(forecast['yhat']<0,0.00001,forecast['yhat'])
        forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.00001,forecast['yhat_lower'])
        forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.00001,forecast['yhat_upper'])
        #model.plot(forecast);

        res = pred[['RMSE','MAE','Accuracy']].groupby([pred['t']]).mean()
        res.index = pd.Series(['Last month','2nd last month','3rd last month']) + '-2019'
        res = pd.DataFrame(res)

        final = forecast[['ds']].iloc[-30:]
        final['weekday'] = final['ds'].dt.weekday
        final['weekday'] = final['weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})
        final[['Price','Low','High']] = forecast[['yhat','yhat_lower','yhat_upper']].iloc[-30:]
        final.set_index('ds',inplace=True)
        
        with open(coin+".html", 'w') as _file:
            _file.write("<h2>Backtest results</h2>"+res.to_html() + "\n\n"+ "<h2>Predictions</h2>" + final.to_html())
        content = get_file(coin+'.html')
        return Response(content, mimetype="text/html")

if __name__ == '__main__':
    app.run(debug=True,threaded=True)