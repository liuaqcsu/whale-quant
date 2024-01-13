#codeing=utf-8
'''2.1 风险敞口与夏普比率
风险敞口是指对风险未采取保护措施导致可能出现损失的那部分资产，本章里指投资股票的资金
夏普比率的公式如下：其中R p R_pR
p
​
 是资产回报，R f R_fR
f
​
 是无风险回报(银行利率或国债利率)，σ p \sigma_pσ
p
​
 是资产回报的标准差。其核心思想是，将一组投资组合的回报率与无风险投资回报率（如银行存款或国债）进行对比，看投资组合的回报会超过无风险投资回报率多少。夏普指数越高，说明投资组合的回报率越高；相反，如果投资组合的回报不及无风险投资的回报，就说明这项投资是不应该进行的
 '''

#codeing=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from utils import wz_data

if __name__ == '__main__':
    #单独实现了去wz网获取数据的接口
    wz = wz_data()
    #股票代码，起始日期，结束日期，这里走的是未复权
    data = wz.get_stock_data_online('601318', '2020-01-01','2020-03-20')
    #返回的直接是一个DataFrame对象
    data.rename(columns={
        '交易时间': 'Date',
        '开盘价': 'Open',
        '最高价': 'High',
        '最低价': 'Low',
        '收盘价': 'Close',
        '成交量': 'Volume',
        '成交额': 'Money',
        '换手率': 'Turnover'
    }, inplace = True)
    data.index = pd.DatetimeIndex(data['Date'])
    print(data.head())
    #创建交易信号
    #新建一个表，使用原表的date作为index
    zgpa_signal = pd.DataFrame(index = data.index)
    #diff字段存收盘价变化
    zgpa_signal['price'] = data['Close']
    zgpa_signal['diff'] = zgpa_signal['price'].diff()
    #NaN数据处理
    zgpa_signal = zgpa_signal.fillna(0.0)
    #股价上涨或不变，标记为0
    #股价下跌，标记为1
    zgpa_signal['signal'] = np.where(zgpa_signal['diff'] >= 0, 0, 1)
    #根据交易信号进行下单
    zgpa_signal['order'] = zgpa_signal['signal'].diff()*100
    zgpa_signal = zgpa_signal.fillna(0.0)
    print(zgpa_signal)
    print('----计算回测数据------')
    #初始资金
    init_cash = 20000
    #持有股票的累加值
    zgpa_signal['hold'] = zgpa_signal['order'].cumsum()
    #单次交易的金额
    zgpa_signal['stock'] = zgpa_signal['order']*zgpa_signal['price']
    #交易金额累加值
    zgpa_signal['cash_cumsum'] = zgpa_signal['stock'].cumsum()
    #剩余现金
    zgpa_signal['cash'] = init_cash - zgpa_signal['cash_cumsum']
    #总资产 = hold*price + cash
    zgpa_signal['total'] = zgpa_signal['hold']*zgpa_signal['price'] + zgpa_signal['cash']
    print(zgpa_signal.head(10))
    #用图形表示 10*6
    #总资产 & 持仓市值
    plt.figure(figsize=(10, 6))
    plt.plot(zgpa_signal['total'], '-', label='总资产')  # 总资产
    plt.plot(zgpa_signal['hold'] * zgpa_signal['price'], '--', label='持仓市值')  # 持仓市值

    plt.grid()
    plt.legend(loc='center right')
    plt.show()
'''2.2 移动平均线及双均线策略
2.2.1　单一移动平均指标
移动平均策略的核心思想非常简单，且十分容易理解

当股价上升且向上穿过N日的均线时，说明股价在向上突破，此时下单买入；当股价下降且向下穿过N日的均线时，说明股价整体出现下跌的趋势，此时下单卖出
当M日均价上升穿过N日的均线（M < N）时，说明股票处于上升的趋势，应下单买入；反之，当M日均价下降且穿过N日均线时，说明股票处于下降的趋势，应下单卖出
实现N日的均线代码如下
 '''
#codeing=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from utils import wz_data

if __name__ == '__main__':
    #单独实现了去wz网获取数据的接口
    wz = wz_data()
    #股票代码，起始日期，结束日期，这里走的是前复权
    data = wz.get_stock_data_online('601318', '2020-01-01','2020-03-20')
    #返回的直接是一个DataFrame对象
    data.rename(columns={
        '交易时间': 'Date',
        '开盘价': 'Open',
        '最高价': 'High',
        '最低价': 'Low',
        '收盘价': 'Close',
        '成交量': 'Volume',
        '成交额': 'Money',
        '换手率': 'Turnover'
    }, inplace = True)
    data.index = pd.DatetimeIndex(data['Date'])
    #print(data.head())
    #计算10天均线
    period = 10
    avg_10 = []
    avg_value = []
    avg_value1 = []
    #保证avg_10最多只有10个数,然后计算均值放入avg_value,初始化的时候值为avg_10[0]
    for price in data['Close']:
        avg_10.append(price)
        if len(avg_10) > 10:
            del avg_10[0]
        avg_value.append(np.mean(avg_10))
    #或者直接用rolling函数计算
    avg_value1 = data['Close'].rolling(10,min_periods=1).mean()
    data = data.assign(avg_10_1 = pd.Series(avg_value, index=data.index))
    data = data.assign(avg_10_2 = pd.Series(avg_value1, index=data.index))
    print(data.head(10))
    # 可视化
    plt.figure(figsize=(10, 6))
    # 绘制股价变化
    plt.plot(data['Close'],  '-', lw=2, c='k', label='价格')
    # 绘制10日均线
    plt.plot(data['avg_10_2'], '--', lw=2, c='b', label='10日均线')
    # 添加图注和网格
    plt.legend()
    plt.grid()
    plt.show()
'''2.2.2 双移动平均策略的实现
       顾名思义，双移动平均策略就是使用两条均线来判断股价未来的走势
       在两条均线中，一条是长期均线（如10日均线），另一条是短期均线（如5日均线）
这种策略基于这样一种假设：股票价格的动量会朝着短期均线的方向移动。当短期均线穿过过长期均线，超过长期移动平均线时，动量将向上，此时股价可能会上涨。然而，如果短期均线的移动方向相反，则股价可能下跌
 '''
def gen_dma_strategy(data):
    strategy = pd.DataFrame(index = data.index)
    #signal交易信号
    strategy['signal'] = 0
    #5日均价
    strategy['avg_5'] = data['Close'].rolling(5,min_periods=1).mean()
    #10日均价
    strategy['avg_10'] = data['Close'].rolling(10,min_periods=1).mean()
    #5日大于10日时signal为1，反之为0
    strategy['signal'] = np.where(strategy['avg_5']>strategy['avg_10'], 1, 0)
    #信号不变不下单，从0到1买入，从1到0卖出
    strategy['order'] = strategy['signal'].diff().fillna(0.0)

    return strategy

strategy = gen_dma_strategy(data)
print(strategy.tail(10))


def draw_dma_strategy(data, strategy):
    # 绘制
    plt.figure(figsize=(10, 5))
    # 绘制股价变化
    plt.plot(data['Close'], lw=2, c='y', label="price")
    plt.plot(strategy['avg_5'], ls='--', c='r', lw=2, label="avg5")
    plt.plot(strategy['avg_10'], ls='-.', c='b', lw=2, label="avg10")

    # 买入卖出信号标记
    plt.scatter(
        strategy.loc[strategy.order == 1].index,
        data['Close'][strategy.order == 1],
        marker='^', s=80, color='r', label='buy'
    )
    plt.scatter(
        strategy.loc[strategy.order == -1].index,
        data['Close'][strategy.order == -1],
        marker='v', s=80, color='g', label='sell'
    )
    #图注，网格，显示
    plt.legend()
    plt.grid()
    plt.show()
draw_dma_strategy(data, strategy)


def back_track(data, strategy, init_cash):
    print('============Test DMA===============')
    # 新建一个positions表，序号和strategy数据表保持一致
    positions = pd.DataFrame(index=strategy.index).fillna(0)
    # 因为A股都是最低100股
    # 因此设置stock字段为交易信号的100倍
    positions['stock'] = strategy['signal'] * 100
    portfolio = pd.DataFrame(index=positions.index)
    portfolio['stock_value'] = positions['stock'].multiply(data['Close'], axis=0)
    # 仓位变化就是下单的数量
    order = positions.diff()
    print('==============交易量===============')
    print(order.tail(20))
    portfolio['stock'] = positions['stock']
    # 初始资金减去下单金额的总和就是剩余的资金
    portfolio['cash'] = init_cash - order.multiply(data['Close'], axis=0).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['stock_value']
    print('==============持有资产===============')
    print(portfolio.tail(10))
    return portfolio
portfolio = back_track(data, strategy, 20000)

def draw_dma_stock_cash(portfolio):
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio['stock_value'], lw=2, ls='--', label='stock_value')
    plt.plot(portfolio['total'], lw=2, ls='-', label='total')
    plt.legend()
    plt.grid()
    plt.show()
draw_dma_stock_cash(portfolio)





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from utils import wz_data
'''海龟策略的其核心要点是：在股价超过过去N个交易日的股价最高点时买入，在股价低于过去N个交易日的股价最低点时卖出。上述的若干个最高点和最低点会组成一个通道，称为“唐奇安通道”'''

def gen_turtle_strategy(data):
    # 创建一个turtle的数据表，使用原始数据的日期序列
    turtle = pd.DataFrame(index=data.index)
    print("----------------turtle------------------")
    # 为什么要用shift(1)要向后移1?
    # 因为要让第六天才能用前5天最高点和最低点
    turtle['price'] = data['Close']
    # 唐其安通道上界=过去5日内的最高价
    turtle['high'] = data['Close'].shift(1).rolling(5).max()
    # 唐其安通道下界=过去5日内的最低价
    turtle['low'] = data['Close'].shift(1).rolling(5).min()
    # 中轨道=0.5*（通道上界+通道下界）
    turtle['mid'] = (turtle['high'] + turtle['low']) / 2
    # 当股价突破下沿时卖出，发出卖出信号，反之买入
    turtle['buy'] = turtle['price'] > turtle['high']
    turtle['sell'] = turtle['price'] < turtle['low']
    return turtle

if __name__ == '__main__':
    #单独实现了去wz网获取数据的接口
    wz = wz_data()
    #股票代码，起始日期，结束日期，这里走的是前复权
    data = wz.get_stock_data_online('601318', '2020-01-01','2020-03-20')
    #返回的直接是一个DataFrame对象
    data.rename(columns={
        '交易时间': 'Date',
        '开盘价': 'Open',
        '最高价': 'High',
        '最低价': 'Low',
        '收盘价': 'Close',
        '成交量': 'Volume',
        '成交额': 'Money',
        '换手率': 'Turnover'
    }, inplace = True)
    data.index = pd.DatetimeIndex(data['Date'])
    print(data.head())
    #实现海龟策略
    turtle_strategy = gen_turtle_strategy(data)
    print(turtle_strategy.head(10))


def gen_turtle_strategy(data):
    # 创建一个turtle的数据表，使用原始数据的日期序列
    turtle = pd.DataFrame(index=data.index)
    print("----------------turtle------------------")
    # 为什么要用shift(1)要向后移1?
    # 因为要让第六天才能用前5天最高点和最低点
    turtle['price'] = data['Close']
    # 唐其安通道上界=过去5日内的最高价
    turtle['high'] = data['Close'].shift(1).rolling(5).max()
    # 唐其安通道下界=过去5日内的最低价
    turtle['low'] = data['Close'].shift(1).rolling(5).min()
    # 中轨道=0.5*（通道上界+通道下界）
    turtle['mid'] = (turtle['high'] + turtle['low']) / 2
    # 当股价突破下沿时卖出，发出卖出信号，反之买入
    turtle['buy'] = data['Close'] > turtle['high']
    turtle['sell'] = data['Close'] < turtle['low']

    # 订单初始状态为0
    turtle['order'] = 0
    # 前一天的仓位
    turtle['preposition'] = 0
    # 当前仓位
    turtle['position'] = 0
    # 初始仓位为0
    position = 0

    # 遍历turtle表
    for k in range(len(turtle)):
        # 当买入信号为true时，且仓位为0时，下单买入一手
        if turtle.buy[k] and position == 0:
            turtle.preposition.values[k] = position  # 保存前一天仓位
            turtle.order.values[k] = 1
            position = 1
            turtle.position.values[k] = position  # 修改当前仓位
        # 卖出信号且持有的时候卖出
        elif turtle.sell[k] and position > 0:
            turtle.preposition.values[k] = position  # 保存前一天仓位
            turtle.order.values[k] = -1
            position = 0
            turtle.position.values[k] = position  # 修改当前仓位
        # 不买不卖时持有仓位
        else:
            turtle.preposition.values[k] = position
            turtle.position.values[k] = position

    return turtle


def draw_turtle_strategy(strategy):
    # 绘制
    plt.figure(figsize=(10, 5))
    # 绘制上沿，下沿，中线
    plt.plot(strategy['price'], lw=2, label="股价")
    plt.plot(strategy['high'], ls='--', c='r', lw=2, label="上沿")
    plt.plot(strategy['low'], ls='-.', c='g', lw=2, label="下沿")
    plt.plot(strategy['mid'], ls='dotted', c='b', lw=2, label="中线")

    # 买入卖出信号标记
    plt.scatter(
        strategy.loc[strategy.order == 1].index,
        strategy['price'][strategy.order == 1],
        marker='^', s=80, color='r', label='买入信号'
    )
    plt.scatter(
        strategy.loc[strategy.order == -1].index,
        strategy['price'][strategy.order == -1],
        marker='v', s=80, color='g', label='卖出信号'
    )
    #图注，网格，显示
    plt.legend()
    plt.grid()
    plt.show()
draw_turtle_strategy(turtle_strategy)


def back_track(strategy, init_cash):
    print('============Test Turtle===============')
    # 新建一个positions表，序号和strategy数据表保持一致
    positions = pd.DataFrame(index=strategy.index).fillna(0.0)
    # 因为A股都是最低100股
    # 因此设置stock字段为交易手的100倍
    positions['stock'] = 100 * strategy['order'].cumsum()
    #资产表
    portfolio = pd.DataFrame(index=positions.index)
    #持仓价值
    portfolio['stock_value'] = positions['stock'].multiply(strategy['price'], axis=0)
    # 仓位变化
    pos_diff = positions.diff()
    print('==============交易量===============')
    portfolio['stock'] = positions['stock']
    # 初始资金减去下单金额的总和就是剩余的资金
    portfolio['cash'] = init_cash - pos_diff.multiply(strategy['price'], axis=0).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['stock_value']
    print('==============持有资产===============')
    print(portfolio.tail(13))
    return portfolio
portfolio = back_track(turtle_strategy, 20000)


def draw_turtle_portfolio(portfolio):
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio['stock_value'], lw=2, ls='--', label='stock_value')
    plt.plot(portfolio['total'], lw=2, ls='-', label='total')
    plt.legend()
    plt.grid()
    plt.show()
draw_turtle_portfolio(portfolio)
'''核心思想如下：如果股价上涨并超过某个点位，说明其上升的动量变强，这时应该买入；反之，则下行的动量变强，此时应该卖出。类似的策略也可以看作基于直觉的交易策略'''