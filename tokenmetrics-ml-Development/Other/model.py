import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
from cryptocompy import coin,price
import tqdm,os
import seaborn as sns
sns.set();

import flask
from flask import request, jsonify, Response, render_template, session, redirect

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
pd.options.display.float_format = '{:,.9f}'.format

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

def supres(low, high, n=28, min_touches=2, stat_likeness_percent=1.5, bounce_percent=5):
    """Support and Resistance Testing
    Identifies support and resistance levels of provided price action data.
    Args:
        n(int): Number of frames to evaluate
        low(pandas.Series): A pandas Series of lows from price action data.
        high(pandas.Series): A pandas Series of highs from price action data.
        min_touches(int): Minimum # of touches for established S&R.
        stat_likeness_percent(int/float): Acceptable margin of error for level.
        bounce_percent(int/float): Percent of price action for established bounce.
    
    ** Note **
        If you want to calculate support and resistance without regard for
        candle shadows, pass close values for both low and high.
    Returns:
        sup(float): Established level of support or None (if no level)
        res(float): Established level of resistance or None (if no level)
    """
    import pandas as pd
    import numpy as np

    # Collapse into dataframe
    df = pd.concat([high, low], keys = ['high', 'low'], axis=1)
    df['sup'] = pd.Series(np.zeros(len(low)))
    df['res'] = pd.Series(np.zeros(len(low)))
    df['sup_break'] = pd.Series(np.zeros(len(low)))
    df['sup_break'] = 0
    df['res_break'] = pd.Series(np.zeros(len(high)))
    df['res_break'] = 0
    
    for x in range((n-1)+n, len(df)):
        # Split into defined timeframes for analysis
        tempdf = df[x-n:x+1]
        
        # Setting default values for support and resistance to None
        sup = None
        res = None
        
        # Identifying local high and local low
        maxima = tempdf.high.max()
        minima = tempdf.low.min()
        
        # Calculating distance between max and min (total price movement)
        move_range = maxima - minima
        
        # Calculating bounce distance and allowable margin of error for likeness
        move_allowance = move_range * (stat_likeness_percent / 100)
        bounce_distance = move_range * (bounce_percent / 100)
        
        # Test resistance by iterating through data to check for touches delimited by bounces
        touchdown = 0
        awaiting_bounce = False
        for y in range(0, len(tempdf)):
            if abs(maxima - tempdf.high.iloc[y]) < move_allowance and not awaiting_bounce:
                touchdown = touchdown + 1
                awaiting_bounce = True
            elif abs(maxima - tempdf.high.iloc[y]) > bounce_distance:
                awaiting_bounce = False
        if touchdown >= min_touches:
            res = maxima
        # Test support by iterating through data to check for touches delimited by bounces
        touchdown = 0
        awaiting_bounce = False
        for y in range(0, len(tempdf)):
            if abs(tempdf.low.iloc[y] - minima) < move_allowance and not awaiting_bounce:
                touchdown = touchdown + 1
                awaiting_bounce = True
            elif abs(tempdf.low.iloc[y] - minima) > bounce_distance:
                awaiting_bounce = False
        if touchdown >= min_touches:
            sup = minima
        if sup:
            df['sup'].iloc[x] = sup
        if res:
            df['res'].iloc[x] = res
    res_break_indices = list(df[(np.isnan(df['res']) & ~np.isnan(df.shift(1)['res'])) & (df['high'] > df.shift(1)['res'])].index)
    for index in res_break_indices:
        df['res_break'].at[index] = 1
    sup_break_indices = list(df[(np.isnan(df['sup']) & ~np.isnan(df.shift(1)['sup'])) & (df['low'] < df.shift(1)['sup'])].index)
    for index in sup_break_indices:
        df['sup_break'].at[index] = 1
    ret_df = pd.concat([df['sup'], df['res'], df['sup_break'], df['res_break']], keys = ['sup', 'res', 'sup_break', 'res_break'], axis=1)
    return ret_df


app = flask.Flask(__name__)
app.config["DEBUG"] = True

#http://localhost:5000/api/predictions/?token=XRP
@app.route('/api/predictions/', methods=['GET'])
def token():
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

        model = Prophet(changepoint_prior_scale=m1,n_changepoints=m2, daily_seasonality=m3,changepoint_range=m4).fit(df)
        data_forecast = df['ds'] + pd.Timedelta(30, unit='days')
        data_forecast = pd.DataFrame(data_forecast)
        data_forecast.columns = ['ds']
        forecast = model.predict(data_forecast)
        forecast['yhat'] = np.where(forecast['yhat']<0,0.000001,forecast['yhat'])
        forecast['yhat_lower'] = np.where(forecast['yhat_lower']<0,0.000001,forecast['yhat_lower'])
        forecast['yhat_upper'] = np.where(forecast['yhat_upper']<0,0.000001,forecast['yhat_upper'])
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
    
# NXT, ZEC, BTS, BTCD, PPC, PRC