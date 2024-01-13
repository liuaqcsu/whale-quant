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
def my_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / sstot

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder

df = pd.DataFrame()
mkt = pd.DataFrame()

#with SSHTunnelForwarder(('206.189.186.74', 22), ssh_password='crypto1234', ssh_username='aagam', remote_bind_address=('127.0.0.1', 3306)) as server:
conn = MySQLdb.connect(host='tokenmetrics.cluster-cxuzrhvtziar.us-east-1.rds.amazonaws.com', user='admin', passwd='WiG8Rled2cTvZ5JibJui',db='tokenmetrics')

cursor = conn.cursor()
    
    #cursor.execute("SELECT * FROM ianbalina.ico_price_daily_summaries;") 
    #m = cursor.fetchone()
query = '''SELECT * FROM ico_price_daily_summaries;'''
data = pd.read_sql_query(query, conn)
df = pd.concat([df,data])

query = '''SELECT * FROM icos;'''
data = pd.read_sql_query(query, conn)
mkt = pd.concat([mkt,data])

conn.close() 


def smoothing(val, alpha):
    temp = []
    temp.append(val[0])
    for i in range(1,len(val)):
        temp.append(alpha * val[i] + (1 - alpha) * temp[-1])
    return temp

def get_target(val, days):
    sign = []
    for i in range(len(val) - days):
        temp = val[i + days] - val[i]
        if temp > 0:
            sign.append(1)
        elif temp < 0:
            sign.append(-1)
        else:
            sign.append(1)
    return sign

def rsi(price, days):
    rel = [0]*days
    gain = [0] * len(price)
    loss = [0] * len(price)
    for i in range(1, len(price)):
        temp = price[i] - price[i-1]
        if temp > 0:
            gain[i] = temp
        else:
            loss[i] = temp
    for i in range(len(price)-days):
        avg_gain = sum(gain[i:i+days])
        avg_loss = sum(loss[i:i+days]) * -1
        if avg_loss == 0:
            avg_loss = 1
        rs = avg_gain / avg_loss
        res = 100 - ((100) / (1 + rs))
        rel.append(res)
    return rel

def stochastic_oscillator(price, days):
    osc = [0]*days
    for i in range(days, len(price)):
        l = min(price[i-days:i]) #low
        h = max(price[i-days:i]) #high
        s = h - l
        if s == 0:
            s = 1
        k = 100 * (price[i] - l) / s
        osc.append(k)
    return osc

def williams_range(price, days):
    osc = [0]*days
    for i in range(days, len(price)):
        l = min(price[i-days:i]) #low
        h = max(price[i-days:i]) #high
        s = h - l
        if s == 0:
            s = 1
        k = -100 * (h - price[i]) / s
        osc.append(k)
    return osc

def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def get_ema(price, days):
    ema = [0]*days
    mul = 2 / (days + 1)
    sma = sum(price[0:days]) / (days)
    ema.append(sma)
    for i in range(days, len(price)):
        sma = sum(price[i-days+1 :i+1]) / days
        temp = (price[i] - ema[-1]) * mul + ema[-1]
        ema.append(temp)
    return ema

def macd(price):
    ''' Moving avg convergence and divergence'''
    signal = []
    ema12 = get_ema(price , 12)[26:]
    ema26 = get_ema(price , 26)[26:]
    ema9 = get_ema(price , 9)[26:]
    
    macd = [a - b for a, b in zip(ema12,ema26)]
    
    for i in range(len(macd)):
        if macd[i] < ema9[i]:
            signal.append(-1)
        elif macd[i] > ema9[i]:
            signal.append(1)
        else:
            signal.append(0)
    
    return [0]*25 + macd

def proc(price, days):
    temp = [0]*days
    for i in range(days, len(price)):
        change = (price[i] - price[i-days]) / price[i-days]
        temp.append(change)
    return temp

def obv(price, vol):
    '''On Balance Volume'''
    temp = [0]
    for i in range(1, len(price)):
        if price[i] > price[i-1]:
            temp.append(vol[i-1] + vol[i])
        elif price[i] < price[i-1]:
            temp.append(vol[i-1] - vol[i])
        else:               # If price not change
            temp.append(vol[i-1])
    return temp

def chng_size(arr, size):
    diff = size - len(arr)
    if len(arr) < size:
        arr = [0]* (size-len(arr)) + arr
    return arr, diff


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


df['rsi30'] = rsi(df['close'],30)
df['rsi45'] = rsi(df['close'],45)
df['rsi60'] = rsi(df['close'],60)

df['sch30'] = stochastic_oscillator(df['close'],30)
df['sch45'] = stochastic_oscillator(df['close'],45)
df['sch60'] = stochastic_oscillator(df['close'],60)

df['w%r30'] = williams_range(df['close'],30)
df['w%r45'] = williams_range(df['close'],45)
df['w%r60'] = williams_range(df['close'],60)

df['macd'] = macd(df['close'])

df['proc30'] = proc(df['close'],30)
df['proc45'] = proc(df['close'],45)
df['proc60'] = proc(df['close'],60)

df['close30'] = [0] * 0 + df['close'].tolist()
df['close45'] = [0] * 15 + df['close'][:-15].tolist()
df['close60'] = [0] * 30 + df['close'][:-30].tolist()


temp = supres(df['low'],df['high'])
temp.fillna(0,inplace=True)
df['sup'] = temp['sup']
df['res'] = temp['res']
df['sup_break'] = temp['sup_break']
df['res_break'] = temp['res_break']


df['month'] = pd.to_datetime(df.index).month
df['year'] = pd.to_datetime(df.index).year
df['week'] = pd.to_datetime(df.index).week
df['weekdays'] = pd.to_datetime(df.index).weekday
df['mon-yr'] = pd.Series(df.index.tolist()).apply(lambda x: str(x)[:7]).tolist()


temp = df['close'].groupby([df['mon-yr']]).mean()
temp = pd.DataFrame(temp)
temp.columns = ['mean-mon-yr']
df = pd.merge(df,temp,on = 'mon-yr', how='left')

temp = df['close'].groupby([df['mon-yr']]).sum()
temp = pd.DataFrame(temp)
temp.columns = ['sum-mon-yr']
df = pd.merge(df,temp,on = 'mon-yr', how='left' )

temp = df['close'].groupby([df['mon-yr']]).quantile(q=0.1)
temp = pd.DataFrame(temp)
temp.columns = ['q1-mon-yr']
df = pd.merge(df,temp,on = 'mon-yr', how='left' )

temp = df['close'].groupby([df['mon-yr']]).quantile(q=0.9)
temp = pd.DataFrame(temp)
temp.columns = ['q9-mon-yr']
df = pd.merge(df,temp,on = 'mon-yr', how='left' )

df = df[60:]
train = df[:-60]
Y_train = df['close'][30:-30]
test = df[-60:-30]
Y_test = df['close'][-30:]

features = train.columns[7:].tolist()
target = ['close']


features.remove('Date')
features.remove('mon-yr')
features += ['close']


#model = CatBoostRegressor(iterations=1000,random_state=0,thread_count=12, eval_metric='R2')
#model.fit(train[features],Y_train, plot=True, verbose=False, eval_set=(test[features], Y_test))

#predictions = model.predict(test[features])
#rmse_ens = np.sqrt(metrics.mean_squared_error(Y_test, predictions)) 

train_stats = X_train.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train[features])
normed_test_data = norm(test[features])

model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(normed_train_data, Y_train, epochs=1000, verbose=0)

predictions = model.predict(normed_test_data)
rmse_ens = np.sqrt(metrics.mean_squared_error(Y_test, predictions)) 

