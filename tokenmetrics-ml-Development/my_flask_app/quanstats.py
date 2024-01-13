import flask
from flask import request, jsonify, Response, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
#import MySQLdb
import os, shutil
import warnings
warnings.filterwarnings("ignore")
import quantstats as qs
qs.extend_pandas()

# source env/bin/activate
#  sudo fuser -k 8000/tcp   -> to kill past processes running on this port
#  gunicorn quanstats:app --daemon &

#http://localhost:5000/api/quantstats/?token=XRP&benchmark=[EOS,BTC]&ratio=[0.5,0.5]
#Api:   http://localhost:5000/api/quantstats/?token=BTC&benchmark=['ETH','BTC','XRP']&ratio=[0.7,0.1,0.2]

def root_dir(): 
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename): 
    try:
        src = os.path.join(root_dir(), filename)
        print(src)
        return open(src).read()
    except IOError as exc:
        return str(exc)

print(True)
app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_SORT_KEYS"] = False

@app.route('/api/quantstats/', methods=['GET'])
def token():
    bar = request.args.to_dict()
    token = bar['token'] + '.html'
    print(token)
    if os.path.exists('reports/'+token):
        #src = os.path.join(root_dir(), 'reports/'+token)
        #print(src)
        content = get_file('reports/'+token)
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
    bar = request.args.to_dict()
    token = bar['token']
    path = 'temp/' + token + '/'
    files = glob.glob(path+'*.csv')
    final = {}
    for file in files:
        x = pd.read_csv(file)
        x = x.to_dict(orient='list')
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
    
@app.route('/api/correlation/', methods=['GET'])
def token3():
	bar = request.args.to_dict()
	token = bar['token']
	temp = pd.read_csv('corr.csv')
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
    bar = request.args.to_dict()
    token = bar['token']

    files = glob.glob('/root/price_prediction/models/'+'*res.csv')
    files = pd.Series(files)
    files = files.str.split('/').str.get(-1).str.split('_').str.get(0)
    files = files.tolist()
    if token not in files:
        return 'Coin Not Available' + "\n\n" + 'Available Coins are: ' + str(files)

    else:
        path1 = '/root/price_prediction/models/' + token + '_final.csv'
        path2 = '/root/price_prediction/models/' + token + '_res.csv'

        final = {}
        x = pd.read_csv(path2)
        x.columns = ['Time','RMSE','MAE','Accuracy']
        x = x.to_dict(orient='list')
        final['result'] = x 

        x = pd.read_csv(path1)
        x['ds'] = pd.to_datetime(x['ds'])
        x = x[x['ds']> d]
        x = x.to_dict(orient='list')
        final['predictions'] = x

        return final

    
if __name__ == '__main__':
    app.run(debug=True,threaded=True)