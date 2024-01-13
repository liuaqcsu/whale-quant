import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from jqdata import *


# 初始化函数，设定基准等等
def initialize(context):
    set_params()

    set_backtest()
    run_daily(trade, 'every_bar')


def set_params():
    g.days = 0
    g.refresh_rate = 5
    g.stocknum = 10


def set_backtest():
    set_benchmark('000001.XSHE')
    set_options('use_real_price', True)
    log.set_level('order', 'error')


def trade(context):
    if g.days % 5 == 0:
        # 这回咱们就把上证50成分股作为股票池
        stocks = get_index_stocks('000016.XSHG')
        # 用query函数获取股票的代码
        q = query(valuation.code,
                  # 还有市值
                  valuation.market_cap,
                  # 净资产，用总资产减去总负债
                  balance.total_assets - balance.total_liability,
                  # 再来一个资产负债率的倒数
                  balance.total_assets / balance.total_liability,
                  # 把净利润也考虑进来
                  income.net_profit,
                  # 还有年度收入增长
                  indicator.inc_revenue_year_on_year,
                  # 研发费用
                  balance.development_expenditure
                  ).filter(valuation.code.in_(stocks))
        # 将这些数据存入一个数据表中
        df = get_fundamentals(q)
        # 给数据表指定每列的列名称
        df.columns = ['code',
                      'mcap',
                      'na',
                      '1/DA ratio',
                      'net income',
                      'growth',
                      'RD']
        # 把股票代码做成数据表的index
        df.index = df['code'].values
        # 然后把原来代码这一列丢弃掉，防止它参与计算
        df = df.drop('code', axis=1)
        # 把除去市值之外的数据作为特征，赋值给X
        X = df.drop('mcap', axis=1)
        # 市值这一列作为目标值，赋值给y
        y = df['mcap']
        # 用0来填补数据中的空值
        X = X.fillna(0)
        y = y.fillna(0)
        # 使用线性回归来拟合数据
        reg = LinearRegression().fit(X, y)
        # 将模型预测值存入数据表
        predict = pd.DataFrame(reg.predict(X),
                               # 保持和y相同的index，也就是股票的代码
                               index=y.index,
                               # 设置一个列名，这个根据你个人爱好就好
                               columns=['predict_mcap'])
        # 使用真实的市值，减去模型预测的市值
        diff = df['mcap'] - predict['predict_mcap']
        # 将两者的差存入一个数据表，index还是用股票的代码
        diff = pd.DataFrame(diff, index=y.index, columns=['diff'])
        # 将该数据表中的值，按生序进行排列
        diff = diff.sort_values(by='diff', ascending=True)
        # 找到市值被低估最多的10只股票
        stockset = list(diff.index[:10])
        sell_list = list(contet.portfolio.positions.keys())

        for stock in sell_list:
            if stock not in stockset[:g.stocknum]:
                stock_sell = stock
                order_target_value(stock_sell, 0)
        if len(context.protfolio.positions) < g.stocknum:
            num = g.stocknum - len(context.portfolio.positions)
            cash = context.portfolio.cash / num
        else:
            cash = 0
            num = 0
        for stock in stockset[:g.stocknum]:
            if stock in sell_list:
                pass
            else:
                stock_buy = stock
                order_target_value(stock_buy, cash)
                num = num - 1
                if num == 0:
                    break
        g.days += 1
    else:
        g.days = g.days + 1