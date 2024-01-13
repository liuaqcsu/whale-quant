# This is a sample Python script.

#coding=utf-8
import sys
import requests
import os
import datetime
import io
import pandas as pd

class wz_data(object):

    def __init__(self):
        self.TOKEN = 'xxxxxx'
        self.DATA_PATH = 'd:/python/data/'

    def get_stock_data_online(self, code, s_date, e_date=None):
        """在线接口，直接下载"""
        #api = 'http://api.waizaowang.com/doc/getStockHSADayKLine?'
        api = 'http://api.waizaowang.com/doc/getHourKLine?type=1&code=000001,000002&ktype=60&startDate=2023-12-10&endDate=2100-01-01&fields=code,name,ktype&export=0&token='
        params = {}

        start_date = s_date
        if not e_date:
            end_date = datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d')
        else:
            end_date = e_date

        params['code'] = code
        params['startDate'] = start_date
        params['endDate'] = end_date
        params['fq'] = '0'
        params['ktype'] = '101'
        params['fields'] = 'tdate,open,high,low,close,cjl,cje,hsl'
        params['export'] = '5'
        params['token'] = self.TOKEN

        r = requests.get(api, params=params).json()
        #df = pd.DataFrame(data=r['data'], columns=r['zh'])
        df = pd.DataFrame(data=r['data'])
        return df
wz = wz_data()
#股票代码，起始日期，结束日期，这里走的是未复权
data = wz.get_stock_data_online('601318', '2023-01-01','2023-03-18')
print(data)


#codeing=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from utils import wz_data

if __name__ == '__main__':
	#之前的代码略
	print('----计算交易信号Signal数据------')
    #创建交易信号字段，命名为Signal
    #如果diff值大于0，则Signal为1，否则为0
    data['Signal'] = np.where(data['diff'] > 0, 1, 0)
    #check
    print(data.head())
    #简单交易策略
    #·当日股价下跌，下一个交易日买入
    #·当日股价上涨，下一个交易日卖出
    #交易信号字段：Signal, diff > 0 Signal=1 卖出，否则Signal=0



#codeing=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from utils import wz_data

if __name__ == '__main__':
	#之前的代码略
	#绘图 画布尺寸10*5
    plt.figure(figsize=(10, 5))
    # 折线图绘制日K线
    data['Close'].plot(linewidth=2, color='k', grid=True)
    # 卖出标志 x轴日期，y轴数值 卖出信号，倒三角
    # matplotlib.pyplot.scatter(x, y, marker, size, color)
    plt.scatter(data['Close'].loc[data.Signal == 1].index,
            data['Close'][data.Signal == 1],
            marker = 'v', s=80, c='g')
    # 买入标志 正三角
    plt.scatter(data['Close'].loc[data.Signal == 0].index,
            data['Close'][data.Signal == 0],
            marker='^', s=80, c='r')
    plt.show()


>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.where(a > 5, 1, -1)
array([-1, -1, -1, -1, -1, -1,  1,  1,  1,  1])
>>> np.where([[True, False], [True, True]],    # 官网上的例子,多维
             [[1, 2], [3, 4]],
             [[9, 8], [7, 6]])
array([[1, 8],
       [3, 4]])


>>> a = 10
>>> np.where([[a > 5, a < 5], [a == 10,a == 7]],
             [["chosen","not chosen"], ["chosen","not chosen"]],
             [["not chosen","chosen"], ["not chosen","chosen"]])

array([['chosen', 'chosen'],
       ['chosen', 'chosen']], dtype='<U10')
