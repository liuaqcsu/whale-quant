#codeing=utf-8
import numpy as np
import pandas as pd
import mplfinance as mpf
from utils import wz_data

if __name__ == '__main__':
    #单独实现了去wz网获取数据的接口
    wz = wz_data()
    #股票代码，起始日期，结束日期，这里走的是未复权
    data = wz.get_stock_data_online('601318', '2020-01-01','2020-03-18')
    #返回的直接是一个DataFrame对象
    data.rename(columns={
        'tdate': 'tdate',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'cjl': 'cjl',
        'cje': 'cje',
        'hsl': 'hsl'
    }, inplace = True)
    #data.index = pd.DatetimeIndex(data['tdate'])
    print(data.head())
    #用.diff()方法来计算每日股价变化情况
    data['diff'] = data['close'].diff()
    print(data.head())
    #绘制蜡烛图
    #type='candle', type='line', type='renko', or type='pnf'
    mpf.plot(data.tail(30), type="candle", volume=True)
