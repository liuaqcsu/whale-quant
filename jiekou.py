import tushare as ts
#pro = ts.pro_api('3337990cd3204d24e06a0b8569ec8d454aa2c056216b958b5e393c05')
#data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code')
ts.set_token('3337990cd3204d24e06a0b8569ec8d454aa2c056216b958b5e393c05')#设置token，只需设置一次
api = ts.pro_api()#初始化接口
#指定一下获取股票数据的起始日期和截止日期
#这里就用2020年1月1日至3月18日的数据
start_date = '2020-01-01'
end_date = '2020-03-18'
#创建数据表，这里选择下载的股票代码为601318
#并把我们把设定的开始日期和截止日期作为参数传入
#pro = ts.pro_api()
data = api.daily(ts_code='000001.SZ', start_date='20230701', end_date='20230718')
#data = api.get_k_data('601318',start = start_date,end = end_date)
data = data.set_index('trade_date')
#下面来检查一下数据表的前5行
data.head()
#给新的字段命名为diff，代表difference
#用.diff()方法来计算每日股价变化情况
data['diff'] = data['close'].diff()
#检查一下前5行
data.head()

#此处会用到numpy，故导入
import numpy as np
#创建交易信号字段，命名为Signal
#如果diff值大于0，则Signal为1，否则为0
data['Signal'] = np.where(data['diff'] > 0, 1, 0)
#检查是否成功
data.head()

#导入画图工具matplotlib
import matplotlib.pyplot as plt
#设置画布的尺寸为10*5
plt.figure(figsize = (10,5))
#使用折线图绘制出每天的收盘价
plt.plot(data['close'],linewidth=2, color='k')
#如果当天股价上涨，标出卖出信号，用倒三角表示
plt.scatter(data['close'].loc[data.Signal==1].index,
        data['close'][data.Signal==1],
        marker = 'v', s=80, c='g')
#如果当天股价下跌给出买入信号，用正三角表示
plt.scatter(data['close'].loc[data.Signal==0].index,
        data['close'][data.Signal==0],
        marker = '^', s=80, c='r')
plt.xticks([0,12,24,36,48])
plt.grid()
#将图像进行展示
plt.show()