import requests
import pandas as pd
import json
import constants.defs as defs
import time
import datetime as dt

from infrastructure.instrument_collection import instrumentCollection as ic
from models.api_price import ApiPrice
from models.open_trade import OpenTrade


LABEL_MAP = {
    'Open': 'o',
    'High': 'h',
    'Low': 'l',
    'Close': 'c',
}

THROTTLE_TIME = 0.3


class OpenFxApi:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(defs.SECURE_HEADER)
        self.last_req_time = dt.datetime.now()

    
    def save_response(self, resp, filename):
        with open(f'./openfx_api/api_data/{filename}.json', 'w') as f:
            d = {}
            d['local_request_date'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            d['response_data'] = resp.json()
            f.write(json.dumps(d, indent=2))

    
    def throttle(self):
        el_s = (dt.datetime.now() - self.last_req_time).total_seconds()
        if el_s < THROTTLE_TIME:
            time.sleep(THROTTLE_TIME - el_s)
        self.last_req_time = dt.datetime.now()

    
    def make_request(self, url, verb='get', code=200, params=None, data=None, headers=None, save_filename="" ):

        self.throttle()

        full_url = f"{defs.OPENFX_URL}/{url}"

        #print(full_url)

        if data is not None:
            data = json.dumps(data)

        try:
            response = None
            if verb == "get":
                response = self.session.get(full_url, params=params, data=data, headers=headers)
            if verb == "post":
                response = self.session.post(full_url, params=params, data=data, headers=headers)
            if verb == "put":
                response = self.session.put(full_url, params=params, data=data, headers=headers)
            if verb == "delete":
                response = self.session.delete(full_url, params=params, data=data, headers=headers)

            #print(response.status_code)
            #print(response.text)
            
            if response == None:
                return False, {'error': 'verb not found'}
            
            if save_filename != "":
                self.save_response(response, save_filename)

            if response.status_code == code:
                return True, response.json()
            else:
                return False, response.json()
            
        except Exception as error:
            return False, {'Exception': error}


    def get_account_summary(self):
        url = f"account"
        ok, data = self.make_request(url, save_filename="account")

        if ok == True:
            return data
        else:
            print("ERROR get_account_summary()", data)
            return None


    def get_account_instruments(self, StatusGroupId='Forex'):
        url = f"symbol"
        ok, symbol_data = self.make_request(url, save_filename="symbol")

        if ok == False:
            print("ERROR get_account_instruments()", symbol_data)
            return None

        target_inst = [x for x in symbol_data if x['StatusGroupId']==StatusGroupId and len(x['Symbol'])==6]
        
        url = f"quotehistory/symbols"
        ok, his_symbol_data = self.make_request(url, save_filename="quotehistory_symbols")

        final_instruments = [x for x in target_inst if x['Symbol'] in his_symbol_data]

        return final_instruments 


    def get_periodicities(self, instrument):
        url = f"quotehistory/{instrument}/periodicities"
        ok, data = self.make_request(url, save_filename="periodicities")

        if ok == True:
            return data
        else:
            print("ERROR get_periodicities()", data)
            return None


    def fetch_candles(self, pair_name, count=-10, granularity="H1", ts_f=None):

        if ts_f is None:
            ts_f = int(pd.Timestamp(dt.datetime.utcnow()).timestamp() * 1000)
            
        params = dict(
            timestamp=ts_f,
            count=count
        )

        if count < 0:
            params['count']=count+1

        base_url = f"quotehistory/{pair_name}/{granularity}/bars/"

        ok_bid, bid_data = self.make_request(base_url+"bid", params=params, save_filename="bids")
        ok_ask, ask_data = self.make_request(base_url+"ask", params=params, save_filename="asks")

        if ok_ask == True and ok_bid == True:
            return True, [ask_data, bid_data]
        
        return False, None
    

    def get_price_dict(self, price_label: str, item):
        data = dict(time=pd.to_datetime(item['Timestamp'], unit='ms'))
        for ohlc in LABEL_MAP.keys():
            data[f"{price_label}_{LABEL_MAP[ohlc]}"]=item[ohlc]
        return data


    def get_candles_df(self, pair_name, count=-10, granularity="H1", date_f=None):

        if date_f is not None:
            date_f = int(pd.Timestamp(date_f).timestamp() * 1000)

        ok, data = self.fetch_candles(pair_name, count=count, granularity=granularity, ts_f=date_f)

        if ok == False:
            return None
        
        data_ask, data_bid = data

        if (data_ask is None) or (data_bid is None):
            return None
        
        if ("Bars" in data_ask == False) or ("Bars" in data_bid == False):
            return pd.DataFrame()
        
        ask_bars = data_ask["Bars"]
        bid_bars = data_bid["Bars"]

        if len(ask_bars) == 0 or len(bid_bars) == 0:
            return pd.DataFrame()

        AvailableTo = pd.to_datetime(data_bid['AvailableTo'], unit='ms')

        bids = [self.get_price_dict('bid', item) for item in bid_bars]
        asks = [self.get_price_dict('ask', item) for item in ask_bars]

        df_bid = pd.DataFrame.from_dict(bids)
        df_ask = pd.DataFrame.from_dict(asks)
        df_merged = pd.merge(left=df_bid, right=df_ask, on='time')    

        for i in ['_o', '_h', '_l', '_c']:
            df_merged[f'mid{i}'] = (df_merged[f'ask{i}'] - df_merged[f'bid{i}']) / 2 + df_merged[f'bid{i}']      

        if count < 0 and df_merged.shape[0] > 0 and df_merged.iloc[-1].time == AvailableTo:
            df_merged = df_merged[:-1]  

        return df_merged


    def last_complete_candle(self, pair_name, granularity):
        df = self.get_candles_df(pair_name, granularity=granularity)
        if df.shape[0] == 0:
            return None
        return df.iloc[-1].time
    
    
    def place_trade(self, pair_name: str, amount: int, direction: int,
                        stop_loss: float=None, take_profit: float=None):
        

        dir_str = "Buy" if direction == defs.BUY else "Sell"

        url = f"trade"

        instrument = ic.instruments_dict[pair_name]
        data = dict(
            Type="Market",
            Symbol=pair_name, 
            Amount=amount,
            Side=dir_str
        )

        if stop_loss is not None:
            data['StopLoss'] = round(stop_loss, instrument.displayPrecision)

        if take_profit is not None:
            data['TakeProfit'] = round(take_profit, instrument.displayPrecision)
            
        print(f"place trade args: {pair_name} {amount} {direction} {stop_loss} {take_profit}")
        print("Place Trade:", data)

        ok, response = self.make_request(url, verb="post", data=data, code=200)

        print(ok, response)

        if 'RemainingAmount' in response and response['RemainingAmount'] != 0:
            ot = self.get_open_trade(response['Id'])

            if ot is not None:
                return response['Id']
            else:
                return None
        else:
            return None


    def get_open_trade(self, trade_id):
            url = f"trade/{trade_id}"
            ok, response = self.make_request(url)

            if ok == True and 'Id' in response:
                return OpenTrade(response)


    def get_open_trades(self):
        url = f"trade"
        ok, response = self.make_request(url)

        if ok == True:
            return [OpenTrade(x) for x in response]
            
    
    def close_trade(self, trade_id):
        url = f"trade"

        params ={
            "trade.type": "Close",
            "trade.id": trade_id
        }

        ok, _ = self.make_request(url, verb="delete", params=params, code=200)

        if ok == True:
            print(f"Closed {trade_id} successfully")
        else:
            print(f"Failed to close {trade_id}")

        return ok


    def get_prices(self, instruments_list):
        url = f"tick/{' '.join(instruments_list)}"

        ok, response = self.make_request(url)

        if ok == True:
            return [ApiPrice(x) for x in response]

        return None


    def get_pip_value(self, instruments_list):
        url = f"pipsvalue"

        params = {
            'targetCurrency': 'EUR',
            'symbols': ' '.join(instruments_list)
        }


        ok, response = self.make_request(url, params=params)

        if ok == True:
            return { x['Symbol']: x['Value'] for x in response }

        return None    
    
    
    def web_api_candles(self, pair_name, granularity, count):

        pair_name = pair_name.replace('_', '')
        df = self.get_candles_df(pair_name, granularity=granularity, count=int(count)*-1)
        if df.shape[0] == 0:
            return None

        cols = ['time', 'mid_o', 'mid_h', 'mid_l', 'mid_c']
        df = df[cols].copy()

        df['time'] = df.time.dt.strftime("%y-%m-%d %H:%M")

        return df.to_dict(orient='list')


















































