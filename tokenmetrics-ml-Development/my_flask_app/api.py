import flask
from flask import request, jsonify, Response, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
import json
#import MySQLdb
import os, shutil
import warnings, datetime
warnings.filterwarnings("ignore")
import quantstats as qs
qs.extend_pandas()
from fbprophet import Prophet
from sklearn import metrics
pd.set_option('float_format', '{:f}'.format)
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

import MySQLdb 
import sshtunnel
from sshtunnel import SSHTunnelForwarder

# source env/bin/activate
#  sudo fuser -k 8000/tcp   -> to kill past processes running on this port
#  gunicorn api:app --daemon &

#http://localhost:5000/api/quantstats/?token=XRP&benchmark=[EOS,BTC]&ratio=[0.5,0.5]
#Api:   http://localhost:5000/api/quantstats/?token=BTC&benchmark=['ETH','BTC','XRP']&ratio=[0.7,0.1,0.2]

def root_dir(): 
    #Getting absolute path of the working directory
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename): 
    #Getting file from the absolute path
    try:
        src = os.path.join(root_dir(), filename)
        print(src)
        return open(src).read()
    except IOError as exc:
        return str(exc)

def cap_floor(x,cap,floor):
    for i in range(len(x)):
        if x[i] > cap:
            x[i] = cap * random.uniform(0.95,1.02)
        elif x[i] < floor:
            x[i] = floor * random.uniform(0.95,1.02)
    return x

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])



def minmax_scaling(array, columns, min_val=0, max_val=1):
    """
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

#df = pd.read_csv('ico_prices_daily.csv')
#temp = df['close'].groupby([df['date'],df['ico_symbol']]).mean()
#temp = temp.unstack().sort_index()

#temp = temp[-90:]
#temp = temp.corr(method='spearman')#.abs()


app = flask.Flask(__name__)    #Flask constructor
app.config["DEBUG"] = True
app.config["JSON_SORT_KEYS"] = False

@app.route('/api/quantstats/', methods=['GET'])
def token():
    '''Function return the quanstats graphs and the other statistics as a webpage'''
    bar = request.args.to_dict()
    token = bar['token'] + '.html'
    print(token)
    if os.path.exists('/home/tokenmetrics/data/quant/reports/'+token):
        #src = os.path.join(root_dir(), 'reports/'+token)
        #print(src)
        content = get_file('/home/tokenmetrics/data/quant/reports/'+token)   # Read the html file of the token
        return Response(content, mimetype="text/html")
    else:
        return 'Token Not Available'
    
    '''
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
        dirpath = os.path.join('temp')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir('temp')
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
            
        #result = result.to_dict()
        c2 = "+".join(benchmark)
        qs.reports.html(toke['returns'],tmp,file='reports.html',c1=bar['token'],c2=c2)
        content = get_file('reports.html')
        return Response(content, mimetype="text/html")'''
    
@app.route('/api/quantstats/data/', methods=['GET'])
def token2():
    '''Function return the data of quanstats graphs and the other statistics as a json'''
    bar = request.args.to_dict()
    token = bar['token']
    path = '/home/tokenmetrics/data/quant/temp/' + token + '/'
    files = glob.glob(path+'*.csv')

    temp = '/home/tokenmetrics/data/quant/temp/' + token + '/' + 'data_key_performance_metrics.csv'
    files.remove(temp)

    final = {}
    for file in files:
        x = pd.read_csv(file)
        x = x.fillna('')
        x = x.to_dict(orient='list')
        file =  file.split('/')[-1][:-4]
        final[file] = x
    return final
    
    '''
    df = pd.read_csv('history.csv')
    df['time'] = df['time'].apply(lambda x: x[:10])
    df.set_index('time',inplace=True)
    df.index = pd.to_datetime(df.index)
    
    bar = request.args.to_dict()
    toke = bar['token']
    benchmark = bar['benchmark'][1:-1].split(',')
    ratio = list(map(float,bar['ratio'][1:-1].split(',')))
    status = True
    coin_not_available = ''
    for con in benchmark:
        if con not in df['coin'].unique().tolist():
            status = False
            coin_not_available = con
    if status == False:
        return coin_not_available + ' Token not available for Benchmark'
    elif toke not in df['coin'].unique().tolist():
        return 'Token not available'
    else:
        dirpath = os.path.join('temp')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir('temp')
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
            
        c2 = "+".join(benchmark)
        qs.reports.html(toke['returns'],tmp,file='reports.html',c1=bar['token'],c2=c2)
        
        files = glob.glob('temp/'+'*.csv')
        str_html = ''
        for file in files:
            res = pd.read_csv(file)
            str_html += "<h2>" + file[5:-4] + "</h2>" + res.to_html() + "\n\n"
        return  '{}'.format(str_html)'''
    
@app.route('/api/quantstats_performance/', methods=['GET'])
def token200():
    '''Function return the data of quanstats performance for particular date and the other statistics as a json'''
    bar = request.args.to_dict()
    token = bar['token']
    date = bar['date']
    
    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]
    
    df = pd.DataFrame()
    icos = pd.DataFrame()

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()
    
    query = '''SELECT * FROM ico_price_daily_summaries where currency='USD';'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])

    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    icos = pd.concat([icos,data])
    conn.close() 

    df = df.drop_duplicates(subset=['ico_symbol','date']).sort_values(by='date')
    df.index = pd.to_datetime(df['date'])

    btc = df['close'][df['ico_symbol'] == 'BTC']
    btc = btc[btc!=0]
    btc = btc.pct_change().fillna(0)

    temp = df['close'][df['ico_symbol'] == token]
    temp = temp[temp!=0]
    temp = temp.pct_change().fillna(0)

    b = btc[btc.index <= date]
    t = temp[temp.index <= date]
    res = qs.reports.metrics(t,benchmark=b, mode='Full', display=False,compounded = True)
    res.columns = [token,'BTC']
    res = res.replace([np.inf, -np.inf], np.nan)
    res = res.fillna('')
    return res.to_dict(orient='index')


@app.route('/api/correlation/', methods=['GET'])
def token3():
    '''Function return the correlation of the coin with other crypto currencies in ascending form'''
    bar = request.args.to_dict()
    token = bar['token']
    temp = pd.read_csv('/home/tokenmetrics/data/corr.csv')
    temp = temp.set_index('ico_symbol')
    if token not in temp.columns.tolist():
        return 'Token not available'
    else:
        result = temp[token].sort_values()
        result = result[result.index != token]
        result = result.dropna()
        result = result.sort_values(ascending=False)
        result = result.to_dict()
        #result = dict(sorted(result.items(), key = lambda x : x[1]))
        return jsonify(result)
    
@app.route('/api/predictions/', methods=['GET'])
def token4():
    '''Function return the predictions on the fly using the prophet model'''
    bar = request.args.to_dict()
    token = str(bar['token'])
    date = bar['date']
    print(token, type(token))
    df = pd.DataFrame()
    
    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    try:
        '''Making connection with the server and reading data'''
        conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
        cursor = conn.cursor()
        query = 'SELECT * FROM ico_price_daily_summaries where ico_symbol ="'+token+'" and currency ="USD";'
        data = pd.read_sql_query(query, conn)
        df = pd.concat([df,data])
        conn.close()

        df = df.drop_duplicates(subset='date').sort_values(by='date')
        df['date'] = pd.to_datetime(df['date'])
        df = df[['close','date']]

        df = df[df['date'] < pd.to_datetime(date)]
        df = df.iloc[-365:]
        df.columns = ['y','ds']
        df = df.sort_values(by='ds') 

        days = 8

        cap = df['y'].iloc[-1] * 1.33
        floor = df['y'].iloc[-1] * 0.80
        df['cap'] = cap
        df['floor'] = floor

        data = df[['y','ds','cap','floor']][:-days]
        true = df['y'][-days:]

        #data = df[['y','ds']][:-days]
        #true = df['y'][-days:]

        x1 = 0
        x2 = 1000000

        for change in [0.9,0.6,0.35,0.1]:  # Backtesting multiple changepoints

            with suppress_stdout_stderr():  # To suppress unwanted output from the models
                model = Prophet(growth='logistic',changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=True,yearly_seasonality=True,
                                    changepoint_range=change).fit(data)

            y = pd.date_range(start=data['ds'].iloc[-1],periods=days+1)
            data_forecast = pd.concat([data['ds'],pd.Series(y)[1:]])

            data_forecast = pd.DataFrame(data_forecast)
            data_forecast.columns = ['ds']
            data_forecast['cap'] = cap
            data_forecast['floor'] = floor
            forecast = model.predict(data_forecast)
            #print(forecast.iloc[-5:])
            y = pd.DataFrame()
            y['True'] = true
            y['Forecasted'] = forecast['yhat'].iloc[-days:].values
            y['date'] = forecast['ds'].iloc[-days:].values
            print(y)
            y = y.dropna()
            y = metrics.mean_absolute_error(y['True'],y['Forecasted'])

            if y<=x2:
                x2 = y
                x1 = change

        '''Using best changepoints for the final model after parametric tuning'''
        with suppress_stdout_stderr():
            model = Prophet(growth='logistic', changepoint_prior_scale=0.7,n_changepoints=25, daily_seasonality=True,yearly_seasonality=True,
                                    changepoint_range=x1).fit(df)

        y = pd.date_range(start=df['ds'].iloc[-1],periods=30)
        data_forecast = pd.concat([df['ds'],pd.Series(y)[1:]])

        data_forecast = pd.DataFrame(data_forecast)
        data_forecast.columns = ['ds']
        data_forecast['cap'] = cap
        data_forecast['floor'] = floor
        forecast = model.predict(data_forecast)

        forecast = forecast.sort_values('ds')
        forecast = forecast.iloc[-30:]
        forecast['yhat'] = cap_floor(forecast['yhat'].values, cap, floor)
        forecast['yhat_lower'] = cap_floor(forecast['yhat_lower'].values, cap*0.8, floor*0.8)
        forecast['yhat_upper'] = cap_floor(forecast['yhat_upper'].values, cap*1.2, floor*1.2)

        #forecast['yhat'] = np.where(forecast['yhat']<floor,floor,forecast['yhat'])
        #forecast['yhat_lower'] = np.where(forecast['yhat_lower']<floor,floor*0.8,forecast['yhat_lower'])
        #forecast['yhat_upper'] = np.where(forecast['yhat_upper']<floor,floor*1.2,forecast['yhat_upper'])

        #forecast['yhat'] = np.where(forecast['yhat']>cap,cap,forecast['yhat'])
        #forecast['yhat_lower'] = np.where(forecast['yhat_lower']>cap,cap*0.8,forecast['yhat_lower'])
        #forecast['yhat_upper'] = np.where(forecast['yhat_upper']>cap,cap*1.2,forecast['yhat_upper'])        
        
        final = forecast[['ds']].iloc[-29:]
        final['weekday'] = final['ds'].dt.weekday
        final['weekday'] = final['weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})
        final[['Price','Low','High']] = forecast[['yhat','yhat_lower','yhat_upper']].iloc[-30:]
        #final.set_index('ds',inplace=True)
        final.columns = ['Date','Weekday','Price','Low','High']
        return final.to_dict(orient='list')
    except:
        final = pd.DataFrame([[],[],[],[],[]]).T
        final.columns = ['Date','Weekday','Price','Low','High']
        return final.to_dict(orient='list')

    
@app.route('/api/fundamental/', methods=['GET'])
def token5():
    '''Passing old and new result, weights, New Grades and Alphs/ROI of fundamentals'''
    d1 = pd.read_csv('/home/tokenmetrics/data/fundamental/New_roi.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/fundamental/Old_roi.csv')
    d3 = pd.read_csv('/home/tokenmetrics/data/fundamental/weights_roi.csv')
    d4 = pd.read_csv('/home/tokenmetrics/data/fundamental/new_grade_roi.csv') 
    d5 = pd.read_csv('/home/tokenmetrics/data/fundamental/alpha_roi_roi.csv')
  

    d1.rename(columns={'Unnamed: 0':'Fundamental_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Fundamental_Grade'},inplace=True)
    d3.rename(columns={'Unnamed: 0':'Fundamental_Grade'},inplace=True)
    d4.rename(columns={'Unnamed: 0':'Technology_Grade'},inplace=True)

    final = {}
    final['old'] = d2.to_dict(orient='list')
    final['new'] = d1.to_dict(orient='list')
    final['weights'] = d3.to_dict(orient='list')
    final['New_Grades'] = d4.to_dict(orient='list')
    final['Alpha_Roi'] = d5.to_dict(orient='list')
    return final

@app.route('/api/technology/', methods=['GET'])
def token6():
    '''Passing old and new result, weights, New Grades and Alphs/ROI of Technology'''
    d1 = pd.read_csv('/home/tokenmetrics/data/technology/New_roi.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/technology/Old_roi.csv') 
    d3 = pd.read_csv('/home/tokenmetrics/data/technology/weights_roi.csv')
    d4 = pd.read_csv('/home/tokenmetrics/data/technology/new_grade_roi.csv')
    d5 = pd.read_csv('/home/tokenmetrics/data/technology/alpha_roi_roi.csv')
   
    d1.rename(columns={'Unnamed: 0':'Technology_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Technology_Grade'},inplace=True)
    d3.rename(columns={'Unnamed: 0':'Technology_Grade'},inplace=True)
    d4.rename(columns={'Unnamed: 0':'Technology_Grade'},inplace=True)


    final = {}
    final['old'] = d2.to_dict(orient='list')
    final['new'] = d1.to_dict(orient='list')
    final['weights'] = d3.to_dict(orient='list')
    final['New_Grades'] = d4.to_dict(orient='list')
    final['Alpha_Roi'] = d5.to_dict(orient='list')
    return final

@app.route('/api/technical/', methods=['GET'])
def token7():
    '''Passing old and new result, weights, New Grades of Technical'''
    d1 = pd.read_csv('/home/tokenmetrics/data/technical/new.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/technical/old.csv')
    d3 = pd.read_csv('/home/tokenmetrics/data/technical/weights.csv')
    d4 = pd.read_csv('/home/tokenmetrics/data/technical/final_grade.csv')
   
    d1.rename(columns={'Unnamed: 0':'Technical_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Technical_Grade'},inplace=True)
    d3.rename(columns={'Unnamed: 0':'Technical_Grade'},inplace=True)
    d4.rename(columns={'Unnamed: 0':'Technical_Grade'},inplace=True)

    final = {}
    final['old'] = d2.to_dict(orient='list')
    final['new'] = d1.to_dict(orient='list')
    final['weights'] = d3.to_dict(orient='list')
    final['New_Grades'] = d4.to_dict(orient='list')
    return final

@app.route('/api/finalgrade/', methods=['GET'])
def token8():
    '''Passing old and new result, weights, New Grades and Alphs/ROI of final grades'''
    d1 = pd.read_csv('/home/tokenmetrics/data/final_grade/New.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/final_grade/Old.csv') 
    d3 = pd.read_csv('/home/tokenmetrics/data/final_grade/weights.csv')
    d4 = pd.read_csv('/home/tokenmetrics/data/final_grade/new_grade.csv')
    d5 = pd.read_csv('/home/tokenmetrics/data/final_grade/alpha_roi.csv')
   
    d1.rename(columns={'Unnamed: 0':'Final_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Final_Grade'},inplace=True)
    d3.rename(columns={'Unnamed: 0':'Final_Grade'},inplace=True)
    d4.rename(columns={'Unnamed: 0':'Final_Grade'},inplace=True)
   
    final = {}
    final['old'] = d2.to_dict(orient='list')
    final['new'] = d1.to_dict(orient='list')
    final['weights'] = d3.to_dict(orient='list')
    final['New_Grades'] = d4.to_dict(orient='list')
    final['Alpha_Roi'] = d5.to_dict(orient='list')
    return final


@app.route('/api/quant_grade/', methods=['GET'])
def token101():
    '''Passing old and new result, weights, New Grades and Alphs/ROI of final grades'''
    d1 = pd.read_csv('/home/tokenmetrics/data/quant_grade/weights.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/quant_grade/new_grade.csv') 
   
    d1.rename(columns={'Unnamed: 0':'Quant_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Quant_Grade'},inplace=True)
   
    final = {}
    final['New_Grades'] = d2.to_dict(orient='list')
    final['weights'] = d1.to_dict(orient='list')
    return final


@app.route('/api/quant_final_grade/', methods=['GET'])
def token102():
    d1 = pd.read_csv('/home/tokenmetrics/data/quant_grade/weights_quant_final.csv')
    d2 = pd.read_csv('/home/tokenmetrics/data/quant_grade/new_grade_quant_final.csv') 
   
    d1.rename(columns={'Unnamed: 0':'Quant_Grade'},inplace=True)
    d2.rename(columns={'Unnamed: 0':'Quant_Grade'},inplace=True)
   
    final = {}
    final['New_Grades'] = d2.to_dict(orient='list')
    final['weights'] = d1.to_dict(orient='list')
    return final


@app.route('/api/investment_stats/', methods=['GET'])
def token9():
    '''Calculate 'all_time_growth_rate','alpha','beta','max_drawdown','time_underwater',
       'sharpe_ratio','sortino_ratio','turnover_rate' of any given coin upto particular date. '''
    bar = request.args.to_dict()
    date = bar['date']    
    icos = bar['ico_id'].split(',')
    price = float(bar['price'])
    print(icos)
    df = pd.DataFrame()
    market_cap = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = '''SELECT * FROM ico_price_daily_summaries;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])
    
    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    market_cap = pd.concat([market_cap,data])

    conn.close()     

    df = df[df['currency'] == 'USD']
    df.set_index('date',inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= pd.to_datetime(date)]
    final = pd.DataFrame()
    for ico_id in icos:
        ico_id = int(ico_id)
        print(ico_id)
        if ico_id in df['ico_id'].unique().tolist():
            print(ico_id)
            bench = df['close'][df['ico_id'] == 3375].drop_duplicates().pct_change().fillna(0)
            returns = df['close'][df['ico_id'] == ico_id].drop_duplicates()#.pct_change().fillna(0)
            returns.iloc[0] = price
            returns = returns.pct_change().fillna(0)

            bench.sort_index(inplace=True)
            returns.sort_index(inplace=True)
            bench = bench[bench.index.isin(returns.index)]
            returns = returns[returns.index.isin(bench.index)]

            try:
                '''We can get 'Alpha','Beta','Max Drawdown ','Longest DD Days','Sharpe','Sortino' directly from quantstats'''
                temp = qs.reports.metrics(returns,benchmark=bench, mode='full', display=False)
                temp.columns = ['Coin','BTC']
                temp = temp['Coin'][['Alpha','Beta','Max Drawdown ','Longest DD Days','Sharpe','Sortino']]
                temp = temp.tolist()

                price = df['close'][df['ico_id'] == ico_id]
                price = (price[-1] - price[0]) / price[0] * 100     # all time growth rate

                df.sort_index(ascending=False, inplace=True)
                vol = df['volume'][df['ico_id'] == ico_id]
                vol = vol.iloc[0]
                cap = market_cap['ico_market_cap'][market_cap['id'] == ico_id]
                #vol = vol.sum() / vol.iloc[-1]

                rate = vol / cap * 100
                rate = rate.iloc[0]   # Turnover rate
                data = pd.DataFrame([price] + temp + [rate], index=['all_time_growth_rate','alpha','beta','max_drawdown','time_underwater',
                                               'sharpe_ratio','sortino_ratio','turnover_rate']).T
                data.index = [ico_id]
                final = pd.concat([data,final])
                print(rate,final)
            except:
                pass
    print(final)
    return final.T.to_dict()

@app.route('/api/stats/', methods=['GET'])
def token12():
    '''Calculate 'all_time_growth_rate','alpha','beta','max_drawdown','time_underwater',
       'sharpe_ratio','sortino_ratio','turnover_rate' of any given coin of all time. '''

    bar = request.args.to_dict()
    date = bar['date']    
    icos = bar['ico_id'].split(',')
    #price = float(bar['price'])
    print(icos)
    df = pd.DataFrame()
    market_cap = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = '''SELECT * FROM ico_price_daily_summaries;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])
    
    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    market_cap = pd.concat([market_cap,data])

    conn.close()     

    df = df[df['currency'] == 'USD']
    df.set_index('date',inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[df.index <= pd.to_datetime(date)]
    final = pd.DataFrame()
    for ico_id in icos:
        ico_id = int(ico_id)
        print(ico_id)
        if ico_id in df['ico_id'].unique().tolist():
            print(ico_id)
            bench = df['close'][df['ico_id'] == 3375].drop_duplicates().pct_change().fillna(0)
            returns = df['close'][df['ico_id'] == ico_id].drop_duplicates()#.pct_change().fillna(0)
            #returns.iloc[0] = price
            returns = returns.pct_change().fillna(0)

            bench.sort_index(inplace=True)
            returns.sort_index(inplace=True)
            bench = bench[bench.index.isin(returns.index)]
            returns = returns[returns.index.isin(bench.index)]

            try:
                temp = qs.reports.metrics(returns,benchmark=bench, mode='full', display=False)
                temp.columns = ['Coin','BTC']
                temp = temp['Coin'][['Alpha','Beta','Max Drawdown ','Longest DD Days','Sharpe','Sortino']]
                temp = temp.tolist()

                price = df['close'][df['ico_id'] == ico_id]
                price = (price[-1] - price[0]) / price[0] * 100

                df.sort_index(ascending=False, inplace=True)
                vol = df['volume'][df['ico_id'] == ico_id]
                vol = vol.iloc[0]
                cap = market_cap['ico_market_cap'][market_cap['id'] == ico_id]
                #vol = vol.sum() / vol.iloc[-1]

                rate = vol / cap * 100
                rate = rate.iloc[0]
                data = pd.DataFrame([price] + temp + [rate], index=['all_time_growth_rate','alpha','beta','max_drawdown','time_underwater',
                                               'sharpe_ratio','sortino_ratio','turnover_rate']).T
                data.index = [ico_id]
                final = pd.concat([data,final])
                print(rate,final)
            except:
                pass
    print(final)
    return final.T.to_dict()


@app.route('/api/grades/', methods=['GET'])
def token10():
    '''Pass all the grades together'''
    fund = pd.read_csv('/home/tokenmetrics/data/fundamental/new_grade_roi.csv')
    tech = pd.read_csv('/home/tokenmetrics/data/technology/new_grade_roi.csv')
    technical = pd.read_csv('/home/tokenmetrics/data/technical/final_grade.csv')
    final = pd.read_csv('/home/tokenmetrics/data/final_grade/new_grade.csv')
    fund = fund[['Coin','Score']]
    fund.columns = ['ico_id','Fundamental Score']
    tech = tech[['Coin','Score']]
    tech.columns = ['ico_id','Technology Score']
    technical.columns = ['ico_id','Technical Score']
    final = final[['ico_id','grade']]
    final.columns = ['ico_id','Final Grade']

    t1 = pd.merge(technical, final)
    t2 = pd.merge(fund,tech)
    t = pd.merge(t1,t2)
    t.set_index('ico_id',inplace=True)
    return t.T.to_dict()

@app.route('/api/percentile_grades/', methods=['GET'])
def token11():
    '''Calculate percentile of 'final_grade','fundamental_grade','technology_grade','technical_grade' by dividing by the max grade '''

    df = pd.DataFrame()
    ieo = pd.DataFrame()
    grade = pd.DataFrame()

    weights = pd.read_csv('/home/tokenmetrics/data/final_grade/weights.csv')

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()
    
    query = '''SELECT * FROM ico_ml_grade_history;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])

    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    ieo = pd.concat([ieo,data])

    query = '''SELECT * FROM ico_grade_history;'''
    data = pd.read_sql_query(query, conn)
    grade = pd.concat([grade,data])

    conn.close()    

    ieos = ieo['id'][ieo['is_traded']==0].unique().tolist()
    grade = grade[grade['ico_id'].isin(ieos)]

    percentile = pd.DataFrame()

    for ico in df['ico_id'].unique().tolist():
        temp = df[df['ico_id'] == ico]
        temp = temp.sort_values('date', ascending=False)
        
        #res = minmax_scaling(temp[['final_grade']], columns=['final_grade'], min_val=temp['final_grade'].min(), max_val=99)
        
        res = temp['final_grade']/df['final_grade'].max() 
        final = res.iloc[0]

        res = temp['fundamental_grade']/df['fundamental_grade'].max()
        fund = res.iloc[0]

        res = temp['technology_grade']/df['technology_grade'].max()
        tech = res.iloc[0]

        res = temp['technical_grade']/df['technical_grade'].max()
        technical = res.iloc[0]

        temp = pd.DataFrame([ico,final,fund,tech,technical])
        percentile = pd.concat([percentile, temp.T], axis=0)


    for ico in grade['ico_id'].unique().tolist():
        temp = grade[grade['ico_id'] == ico]
        temp = temp.sort_values('date', ascending=False)
        
        
        weighted_ieo = temp['tokenmetrics_score'].iloc[0]*float(weights.iloc[-1][1]) + temp['tech_scorecard_score'].iloc[0]*float(weights.iloc[-1][2])
        max_ieo = 100*float(weights.iloc[-1][1]) + 100*float(weights.iloc[-1][2])
        final = weighted_ieo / max_ieo

        
        #res = temp['grade']/df['final_grade'].max() 
        #final = res.iloc[0]

        res = temp['tokenmetrics_score']/100 #df['fundamental_grade'].max()
        fund = res.iloc[0]

        res = temp['tech_scorecard_score']/100  #df['technology_grade'].max()
        tech = res.iloc[0]

        temp = pd.DataFrame([ico,final,fund,tech,0])
        percentile = pd.concat([percentile, temp.T], axis=0)


    percentile.columns = ['ico_id','final_grade','fundamental_grade','technology_grade','technical_grade']
    percentile.set_index('ico_id', inplace=True)
    percentile = percentile*100
    percentile[percentile > 100] = 100
    return percentile.T.to_dict() 

def get_holdings(start,portfolio_amount, percentile = 'False', grade_type='final_grade'):
    df = pd.DataFrame()
    price = pd.DataFrame()
    ieo = pd.DataFrame()
    
    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()
    
    if percentile == 'TRUE':
        query = '''SELECT * FROM ico_ml_percentile_grade_history;'''
        data = pd.read_sql_query(query, conn)
        df = pd.concat([df,data])    
    else:
        query = '''SELECT * FROM ico_ml_grade_history;'''
        data = pd.read_sql_query(query, conn)
        df = pd.concat([df,data])
    
    query = 'SELECT * FROM ico_price_daily_summaries where date ="'+start+'" and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    price = pd.concat([price,data])
    
    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    ieo = pd.concat([ieo,data])
    conn.close() 
    
    ieos = ieo['id'][ieo['is_traded']==0].unique().tolist()

    ieo = ieo[['id','ico_market_cap']]
    temp = price[['ico_id','volume']]
    temp = pd.merge(ieo,temp, left_on = 'id',right_on = 'ico_id')
    temp['turnover_rate'] = temp['volume'] / temp['ico_market_cap']
    ids = temp['id'][temp['ico_market_cap']<500000].tolist()
    ids = ids + temp['id'][temp['turnover_rate'] < 0.1].tolist()
    ieos = ieos + ids

    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] == start]
    df = df[~df['ico_id'].isin(ieos)]
    df = df.sort_values(by=grade_type, ascending=False)
    df = df[:20]
    price = price[['ico_id','close']]
    df = df[['ico_id','date',grade_type]]
    final = pd.merge(df,price,on='ico_id')
    final = final[:10]
    final.columns = ['ico_id','date','Grade','Initial Price']
    
    final['Grade Weight'] = final['Grade'] / final['Grade'].sum()
    final['Holding Amount'] = final['Grade Weight'] * portfolio_amount
    final['Current Holding Amount'] = final['Holding Amount']
    final['Holding Volume'] = final['Holding Amount'] / final['Initial Price']
    return final


def portfolio_amount(date,final):
    price = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = 'SELECT * FROM ico_price_daily_summaries where date ="'+date+'" and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    price = pd.concat([price,data])

    conn.close() 
    
    price = price[['ico_id','close']]
    price.columns = ['ico_id','New Price']
    
    temp = pd.merge(final,price)
    #print(date, final, price, 'Amount', temp['Holding Volume'] * temp['New Price'])
    return temp['Holding Volume'] * temp['New Price']



def get_hold(date, holdings,percentile, grade_type, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date']).dt.month[0] != datetime.today().month:      #datetime.today().date() == datetime.today().date().replace(day=1):
        amount = portfolio_amount(date, holdings)
        holdings = get_holdings(date,sum(amount),percentile, grade_type)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100

@app.route('/api/monthly_index/', methods=['GET'])
def token13():
    bar = request.args.to_dict()
    grade_type = str(bar['grade_type'])
    percentile = str(bar['percentile'])

    if percentile == 'TRUE':
        path = '/home/tokenmetrics/data/indexes/monthly/'+'percentile/'+grade_type+'_holdings.csv'
    else:
        path = '/home/tokenmetrics/data/indexes/monthly/'+grade_type+'_holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_hold(date, holdings, percentile, grade_type, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_holdings('2020-04-05',10000, percentile, grade_type)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Grade','Initial Price','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Grade','Initial Price','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings

def get_weekly_hold(date, holdings,percentile, grade_type, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date'].iloc[0]) + pd.to_timedelta(7,unit='days') <= datetime.today().date():
        amount = portfolio_amount(date, holdings)
        holdings = get_holdings(date,sum(amount),percentile, grade_type)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100

@app.route('/api/weekly_index/', methods=['GET'])
def token14():
    bar = request.args.to_dict()
    grade_type = str(bar['grade_type'])
    percentile = str(bar['percentile'])

    if percentile == 'TRUE':
        path = '/home/tokenmetrics/data/indexes/weekly/'+'percentile/'+grade_type+'_holdings.csv'
    else:
        path = '/home/tokenmetrics/data/indexes/weekly/'+grade_type+'_holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_weekly_hold(date, holdings,percentile, grade_type, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_holdings('2020-04-05',10000,percentile, grade_type)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Grade','Initial Price','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Grade','Initial Price','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings


def get_quarterly_hold(date, holdings,percentile, grade_type, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date'].iloc[0]) + pd.to_timedelta(90,unit='days') <= datetime.today().date():
        amount = portfolio_amount(date, holdings)
        holdings = get_holdings(date,sum(amount),percentile, grade_type)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100


@app.route('/api/quarterly_index/', methods=['GET'])
def token15():
    bar = request.args.to_dict()
    grade_type = str(bar['grade_type'])
    percentile = str(bar['percentile'])

    if percentile == 'TRUE':
        path = '/home/tokenmetrics/data/indexes/quarterly/'+'percentile/'+grade_type+'_holdings.csv'
    else:
        path = '/home/tokenmetrics/data/indexes/quarterly/'+grade_type+'_holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_quarterly_hold(date, holdings, percentile, grade_type, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_holdings('2020-04-05',10000,percentile, grade_type)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Grade','Initial Price','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Grade','Initial Price','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings


def get_yearly_hold(date, holdings,percentile, grade_type, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date'].iloc[0]) + pd.to_timedelta(365,unit='days') <= datetime.today().date():
        amount = portfolio_amount(date, holdings)
        holdings = get_holdings(date,sum(amount),percentile, grade_type)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100


@app.route('/api/yearly_index/', methods=['GET'])
def token16():
    bar = request.args.to_dict()
    grade_type = str(bar['grade_type'])
    percentile = str(bar['percentile'])

    if percentile == 'TRUE':
        path = '/home/tokenmetrics/data/indexes/yearly/'+'percentile/'+grade_type+'_holdings.csv'
    else:
        path = '/home/tokenmetrics/data/indexes/yearly/'+grade_type+'_holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_yearly_hold(date, holdings, percentile, grade_type, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_holdings('2020-04-05',10000,percentile, grade_type)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Grade','Initial Price','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Grade','Initial Price','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings


def get_daily_hold(date, holdings,percentile, grade_type, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date'].iloc[0]) + pd.to_timedelta(1,unit='days') <= datetime.today().date():
        amount = portfolio_amount(date, holdings)
        holdings = get_holdings(date,sum(amount),percentile, grade_type)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100


@app.route('/api/daily_index/', methods=['GET'])
def token160():
    bar = request.args.to_dict()
    grade_type = str(bar['grade_type'])
    percentile = str(bar['percentile'])

    if percentile == 'TRUE':
        path = '/home/tokenmetrics/data/indexes/daily/'+'percentile/'+grade_type+'_holdings.csv'
    else:
        path = '/home/tokenmetrics/data/indexes/daily/'+grade_type+'_holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_daily_hold(date, holdings, percentile, grade_type, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_holdings('2020-05-28',10000,percentile, grade_type)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Grade','Initial Price','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Grade','Initial Price','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings


def get_accuracy(symbol):
    pred = pd.DataFrame()
    true = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = 'SELECT * FROM ico_price_daily_summaries where ico_id ="'+ str(symbol) + '"and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    true = pd.concat([true,data])
    
    ico_id = true['ico_id'].iloc[0]

    query = 'SELECT * FROM predictions where ico_id="'+ str(ico_id) + '";'
    data = pd.read_sql_query(query, conn)
    pred = pd.concat([pred,data])
    conn.close() 

    pred = pred[pred['ico_id'] == ico_id]
    pred['month'] = pred['predicted_date'].apply(lambda x: str(x)[:7])
    true['month'] = true['date'].apply(lambda x: str(x)[:7])
    final = pd.merge(pred[['ico_id','predicted_date','price']],true[['ico_id','date','close','month']], 
         left_on = ['ico_id','predicted_date'],right_on = ['ico_id','date'])

    metric = pd.DataFrame()
    for mont in final['month'].unique().tolist():
        temp = final[final['month'] == mont]
        mae = mean_absolute_error(temp['close'], temp['price'])
        mse = mean_squared_error(temp['close'], temp['price'])

        score = []
        for i in range(len(temp)):
            x = abs(temp['price'].iloc[i] - temp['close'].iloc[i]) / temp['close'].iloc[i]
            score.append(1-x)

        te = pd.DataFrame([mont,round(mae,2),round(mse,2),round(np.sqrt(mse),2), round(sum(score) / len(score) , 2)])
        metric = pd.concat([metric,te.T])

    metric.columns = ['Months','MAE','MSE','RMSE','Accuracy']
    metric.set_index('Months', inplace=True)
    return metric


def get_prediction_holdings(start,portfolio_amount, days = 30):
    df = pd.DataFrame()
    price = pd.DataFrame()
    grade = pd.DataFrame()
    ieo = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = 'SELECT * FROM ico_price_daily_summaries where date ="'+start+'" and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    price = pd.concat([price,data])

    query = '''SELECT * FROM predictions;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])

    query = '''SELECT * FROM ico_ml_percentile_grade_history;'''
    data = pd.read_sql_query(query, conn)
    grade = pd.concat([grade,data])

    query = '''SELECT * FROM icos;'''
    data = pd.read_sql_query(query, conn)
    ieo = pd.concat([ieo,data])
    conn.close() 
    

    ieos = ieo['id'][ieo['is_traded']==0].unique().tolist()

    ieo = ieo[['id','ico_market_cap']]
    temp = price[['ico_id','volume']]
    temp = pd.merge(ieo,temp, left_on = 'id',right_on = 'ico_id')
    temp['turnover_rate'] = temp['volume'] / temp['ico_market_cap']
    ids = temp['id'][temp['ico_market_cap']<500000].tolist()
    ids = ids + temp['id'][temp['turnover_rate'] < 0.1].tolist()
    ieos = ieos + ids

    df = df[~df['ico_id'].isin(ieos)]

    if days == 30:
        pred_date = pd.to_datetime(start) + pd.to_timedelta(28,unit='D') #df['predicted_date'].sort_values(ascending=False).iloc[0]
    elif days == 7:
        pred_date = pd.to_datetime(start) + pd.to_timedelta(7,unit='D')
    df = df[df['predicted_date'] == pred_date]
    df = pd.merge(df,price[['ico_id','close']], on='ico_id')
    df['growth'] = (df['price'] - df['close']) / df['close']
    df = df.sort_values('growth', ascending=False)
    final = df[['ico_id','growth','close']]

    grade = grade[grade['date'] == pd.to_datetime(start)]
    grade = grade[['ico_id','final_grade']]
    final = pd.merge(final,grade)
    final = final.sort_values('growth',ascending=False)
    
   
    accuracy_filter = []
    accuracy = []
    count = 0
    for ico_id in final['ico_id'][:50]:
        acc = get_accuracy(ico_id)
        if sum(acc['Accuracy'][-3:] > 0.8) == 3:
            accuracy_filter.append(True)
            count += 1
        else:
            accuracy_filter.append(False)
        accuracy.append(acc['Accuracy'][-3:].tolist())

        if count >= 10:
            break;
    
    final = final[:len(accuracy_filter)]
    final['accuracy'] = accuracy
    final['accuracy_filter'] = accuracy_filter
    final = final[final['accuracy_filter'] == True]
    final = final[:10]    

    final['date'] = start
    final = final[['ico_id','date','growth','final_grade','close','accuracy']]
    
    final.columns = ['ico_id','date','Growth','Grade','Initial Price','Accuracy']
    final['Grade Weight'] = final['Grade'] / final['Grade'].sum()
    final['Holding Amount'] = final['Grade Weight'] * portfolio_amount
    final['Current Holding Amount'] = final['Holding Amount']
    final['Holding Volume'] = final['Holding Amount'] / final['Initial Price']
    return final


def get_predicted_monthly_hold(date, holdings, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date']).dt.month[0] != datetime.today().month:      #datetime.today().date() == datetime.today().date().replace(day=1):
        date = str(datetime.today().date().replace(day=1))
        amount = portfolio_amount(date, holdings)
        holdings = get_prediction_holdings(date,sum(amount),30)
        print('Amount',sum(amount), holdings)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100

@app.route('/api/predicted_monthly_index/', methods=['GET'])
def token17():
    path = '/home/tokenmetrics/data/indexes/predicted/monthly/holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_predicted_monthly_hold(date, holdings, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_prediction_holdings('2020-07-15',51697.78, 30)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Growth','Grade','Initial Price','Accuracy','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Growth','Grade','Initial Price','Accuracy','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings

def get_predicted_weekly_hold(date, holdings, path):
    # If initial holdings present
    if pd.to_datetime(holdings['date'].iloc[0]) + pd.to_timedelta(7,unit='days') <= datetime.today().date():
        amount = portfolio_amount(date, holdings)
        holdings = get_prediction_holdings(date,sum(amount),7)
        if len(holdings) > 8:
            holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100
    else:
        amount = portfolio_amount(date, holdings)
        return amount, sum(amount) / 100

@app.route('/api/predicted_weekly_index/', methods=['GET'])
def token18():
    path = '/home/tokenmetrics/data/indexes/predicted/weekly/holdings.csv'

    date = str(datetime.today().date())

    if os.path.isfile(path):
        holdings = pd.read_csv(path)
        cur_hold, index = get_predicted_weekly_hold(date, holdings, path)
        holdings = pd.read_csv(path)
        holdings['Current Holding Amount'] = cur_hold
        holdings.to_csv(path, index=False)
    else:
        holdings = pd.DataFrame()
        holdings = get_prediction_holdings(date,10000, 7)
        holdings.to_csv(path, index=False)
        amount = portfolio_amount(date, holdings)
        index = sum(amount) / 100

    holdings = pd.read_csv(path)
    holdings['date'] = holdings['date'].astype('str')
    holdings.set_index('ico_id', inplace=True)
    holdings = holdings[['date','Growth','Grade','Initial Price','Accuracy','Grade Weight','Holding Amount','Current Holding Amount','Holding Volume']]
    holdings.columns = ['date','Growth','Grade','Initial Price','Accuracy','Grade Weight','Initial Cash Value','Current Cash Value','Amount of Tokens']
    holdings = holdings[holdings['Grade'].notnull()]
    holdings = holdings.T.to_dict()
    holdings['Index Value'] = index
    return holdings


@app.route('/api/metrics/', methods=['GET'])
def token20():
    bar = request.args.to_dict()
    symbol = str(bar['symbol'])
    pred = pd.DataFrame()
    true = pd.DataFrame()

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = '''SELECT * FROM predictions;'''
    data = pd.read_sql_query(query, conn)
    pred = pd.concat([pred,data])

    query = 'SELECT * FROM ico_price_daily_summaries where ico_symbol ="'+ symbol + '"and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    true = pd.concat([true,data])
    conn.close() 

    ico_id = true['ico_id'].iloc[0]

    if pred['ico_id'].isin([ico_id]).sum() != 0:
        pred = pred[pred['ico_id'] == ico_id]
        pred['month'] = pred['predicted_date'].apply(lambda x: str(x)[:7])
        true['month'] = true['date'].apply(lambda x: str(x)[:7])
        final = pd.merge(pred[['ico_id','predicted_date','price']],true[['ico_id','date','close','month']], left_on = ['ico_id','predicted_date'],right_on = ['ico_id','date'])

        metric = pd.DataFrame()
        for mont in final['month'].unique().tolist():
            temp = final[final['month'] == mont]
            mae = mean_absolute_error(temp['close'], temp['price'])
            mse = mean_squared_error(temp['close'], temp['price'])
   
            score = []
            for i in range(len(temp)):
                x1 = temp['price'].iloc[i]
                x2 = temp['close'].iloc[i]
                if x1 > x2:
                    x = abs(x1 - x2) / x1
                else:
                    x = abs(x2 - x1) / x2
                #x = abs(temp['price'].iloc[i] - temp['close'].iloc[i]) / temp['close'].iloc[i]
                score.append(1-x)
    
            te = pd.DataFrame([mont,round(mae,2),round(mse,2),round(np.sqrt(mse),2), round(sum(score) / len(score) , 2)])
            metric = pd.concat([metric,te.T])

        metric.columns = ['Months','MAE','MSE','RMSE','Accuracy']
        metric.set_index('Months', inplace=True)
        return metric.to_dict()

    else:
        return ''

@app.route('/api/growth/', methods=['GET'])
def token21():
    date = str(datetime.today().date())
    price = pd.DataFrame()
    df = pd.DataFrame()
    
    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = 'SELECT * FROM ico_price_daily_summaries where date ="'+date+'" and currency ="USD";'
    data = pd.read_sql_query(query, conn)
    price = pd.concat([price,data])

    query = '''SELECT * FROM predictions;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])
    
    conn.close()

    df.sort_values('predicted_date', ascending=False, inplace=True)
    df = df[df['predicted_date'] == df['predicted_date'].iloc[0]]
    df = pd.merge(df, price[['ico_id','close']])
    df['Growth'] = (df['price'] - df['close']) / df['close'] 
    df = df[['ico_id','Growth']]
    df = df.set_index('ico_id')
    df = df.to_dict()
    return df['Growth']


def get_index_stat(p,q,r,df,btc):
    data = df[df['investor_type'] == p][df['investment_style'] == q][df['time_horizon'] == r]
    if len(data) > 1:
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        data['index'] = data['index'].astype('float')

        eoy_btc = (btc['close'].iloc[-1] - btc['close'].iloc[0]) / btc['close'].iloc[0] * 100
        eoy_index = (data['index'].iloc[-1] - data['index'].iloc[0]) / data['index'].iloc[0] * 100

        btc = btc['close'].pct_change().fillna(0)

        data.set_index('date', inplace=True)
        data = data['index'].astype('float').pct_change().fillna(0)
        print(eoy_index,eoy_btc,len(data))

        temp = qs.reports.metrics(data,benchmark=btc, mode='full', display=False)
        temp.columns = ['Coin','BTC']

        tem = pd.DataFrame(index=['EOY Return Vs Benchmark'],data = [[eoy_index,eoy_btc]])
        tem.columns = ['Coin','BTC']
        temp = pd.concat([temp,tem])
        
        temp = temp.replace([np.inf, -np.inf], np.nan)
        temp = temp.fillna(0)
        return temp
    else:
        return ''


@app.route('/api/index_stats/', methods=['GET'])
def token22():
    bar = request.args.to_dict()
    p = str(bar['investor_type'])
    q = str(bar['investment_style'])
    r = str(bar['time_horizon'])

    df = pd.DataFrame()
    price = pd.DataFrame()   

    cred = pd.read_table('/home/tokenmetrics/credentials.txt',sep=',')
    cred = cred.to_dict('records')[0]

    conn = MySQLdb.connect(host=cred['host'], user=cred['user'], passwd=cred['passwd'], db='tokenmetrics')
    cursor = conn.cursor()

    query = '''SELECT * FROM indice_values;'''
    data = pd.read_sql_query(query, conn)
    df = pd.concat([df,data])

    query = '''SELECT * FROM ico_price_daily_summaries where ico_id = 3375 and currency = "USD";'''
    data = pd.read_sql_query(query, conn)
    price = pd.concat([price,data])

    conn.close() 

    i = df['investor_type'].unique().tolist()
    j = df['investment_style'].unique().tolist()
    k = df['time_horizon'].unique().tolist()

    price['date'] = pd.to_datetime(price['date'])
    price = price.sort_values('date')
    df = df.sort_values('date')
    btc = price[price['date'] >= pd.to_datetime(df['date'].iloc[0])]
    btc.set_index('date', inplace=True)

    quan_stat = get_index_stat(p,q,r,df, btc)
    if len(quan_stat) > 0:
        return quan_stat.to_dict()
    else:
        return ''

    #final = pd.DataFrame()
    #for p in i:
    #    for q in j:
    #        for r in k:
    #            quan_stat = get_index_stat(p,q,r,df, btc)
    #            if len(quan_stat)>0:
    #                quan_stat['investor_type'] = str(p)
    #                quan_stat['investment_style'] = str(q)
    #                quan_stat['time_horizon'] = str(r)
    #                final = pd.concat([final,quan_stat])

    #final['Metric'] = final.index
    #return str(final.set_index(['investor_type', 'investment_style','time_horizon','Metric']).to_dict())


if __name__ == '__main__':
    app.run(debug=True,threaded=True)