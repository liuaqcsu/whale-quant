import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import MySQLdb
import warnings
warnings.filterwarnings("ignore")

#Api:   http://localhost:5000/api/correlation/?token=BTC
# Enter token in last as BTC
#host = ''
#user = ''
#password = ''
#db = ''
'''
def get_data(host,user,password,db):
	conn = MySQLdb.connect(host=host, user=user, passwd=password, db=db) # Make connection
	print('Connection Established')
	query = "SELECT * FROM ico_price_daily_summaries"
	df = pd.read_sql_table(query, conn)
	return df
'''
#df = get_data(host,user,password,db)
df = pd.read_csv('ico_price_daily_summaries.csv')
temp = df['close'].groupby([df['date'],df['ico_symbol']]).mean()
temp = temp.unstack().sort_index()

#temp = temp[-90:]
temp = temp.corr(method='spearman').abs()

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_SORT_KEYS"] = False

@app.route('/api/correlation/', methods=['GET'])
def token():
	bar = request.args.to_dict()
	token = bar['token']
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

if __name__ == '__main__':
    app.run(debug=True,threaded=True)