import requests
import pandas as pd
import json
from dateutil import parser
#API_KEY='aaaa9999bbbb8888cccc7777dddd6666-eeee5555ffff4444aaaa3333bbbb2222'
API_KEY = "40e33254c833d446edd42d6f250beb7c-d81a8336a460b8c8c3f68ca02b3b29ad"
ACCOUNT_ID = "101-012-17790649-001"
OANDA_URL = "https://api-fxpractice.oanda.com/v3"
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
})
params = dict(
    count = 10,
    granularity = "H1",
    price = "MBA"
)
url = f"{OANDA_URL}/instruments/EUR_USD/candles"
response = session.get(url, params=None, data=None, headers=None)
response.status_code
data = response.json()


instruments_list = data['instruments']
len(instruments_list)
instruments_list[0].keys()
key_i = ['name', 'type', 'displayName', 'pipLocation',
         'displayPrecision', 'tradeUnitsPrecision', 'marginRate']
instruments_dict = {}
for i in instruments_list:
    key = i['name']
    instruments_dict[key] = { k: i[k] for k in key_i }
instruments_dict['USD_CAD']
pow(10, -4)
with open("../data/instruments.json", "w") as f:
    f.write(json.dumps(instruments_dict, indent=2))


def fetch_candles(pair_name, count=10, granularity="H1"):
    url = f"{OANDA_URL}/instruments/{pair_name}/candles"
    params = dict(
        count=count,
        granularity=granularity,
        price="MBA"
    )
    response = session.get(url, params=params, data=None, headers=None)
    data = response.json()

    if response.status_code == 200:
        if 'candles' not in data:
            data = []
        else:
            data = data['candles']
    return response.status_code, data


def get_candles_df(data):
    if len(data) == 0:
        return pd.DataFrame.empty

    prices = ['mid', 'bid', 'ask']
    ohlc = ['o', 'h', 'l', 'c']

    final_data = []
    for candle in data:
        if candle['complete'] == False:
            continue
        new_dict = {}
        new_dict['time'] = parser.parse(candle['time'])
        new_dict['volume'] = candle['volume']
        for p in prices:
            for o in ohlc:
                new_dict[f"{p}_{o}"] = float(candle[p][o])
        final_data.append(new_dict)
    df = pd.DataFrame.from_dict(final_data)
    return df


def create_data_file(pair_name, count=10, granularity="H1"):
    code, data = fetch_candles(pair_name, count, granularity)
    if code != 200:
        print("Failed", pair_name, data)
        return
    if len(data) == 0:
        print("No candles", pair_name)
    candles_df = get_candles_df(data)
    candles_df.to_pickle(f"../data/{pair_name}_{granularity}.pkl")
    print(f"{pair_name} {granularity} {candles_df.shape[0]} candles, {candles_df.time.min()} {candles_df.time.max()}")


code, data = fetch_candles("EUR_USD", count=10, granularity="H4")
candles_df = get_candles_df(data)
create_data_file("EUR_USD", count=10, granularity="H4")
our_curr = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'NZD', 'CAD', 'AUD']
for p1 in our_curr:
    for p2 in our_curr:
        pr = f"{p1}_{p2}"
        if pr in instruments_dict:
            for g in ["H1", "H4"]:
                create_data_file(pr, count=4001, granularity=g)