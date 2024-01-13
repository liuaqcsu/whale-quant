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