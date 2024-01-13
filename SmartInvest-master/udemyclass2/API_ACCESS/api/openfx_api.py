import requests
import pandas as pd
import json
import sys
import os
#module_path = os.path.abspath('/Users/stella/Downloads/SmartInvest-master/udemyclass2/API_ACCESS/constants')
#sys.path.append(module_path)
import constants.defs as defs
#import defs
#from constants.defs import defs
#import constants.defs
import time
import datetime as dt

class defs:
    OPENFX_URL = "https://marginalttdemowebapi.fxopen.net:8443/api/v2"

    API_ID = "179518de-64a7-4451-ade1-589b20c43001"
    API_KEY = "293nWE4Mme3N3Y9R"
    API_SECRET = "QxNm8AAEbYK6BB6A2XANCGmnZbG6GCSnpwQfD8d4zHEzrMHXNXpsb6bgjAx73E8H"

    SECURE_HEADER = {
        "Authorization": f"Basic {API_ID}:{API_KEY}:{API_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


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

    def make_request(self, url, verb='get', code=200, params=None, data=None, headers=None, save_filename=""):

        self.throttle()

        full_url = f"{defs.OPENFX_URL}/{url}"

        # print(full_url)

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

            # print(response.status_code)
            # print(response.text)

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

        target_inst = [x for x in symbol_data if x['StatusGroupId'] == StatusGroupId and len(x['Symbol']) == 6]

        url = f"quotehistory/symbols"
        ok, his_symbol_data = self.make_request(url, save_filename="quotehistory_symbols")

        final_instruments = [x for x in target_inst if x['Symbol'] in his_symbol_data]

        return final_instruments