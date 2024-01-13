from api.oanda_api import OandaApi
from infrastructure.instrument_collection import instrumentCollection

if __name__ == '__main__':
    api = OandaApi()
    
    instrumentCollection.CreateFile(api.get_account_instruments(), "./data")
    instrumentCollection.LoadInstruments("./data")
    instrumentCollection.PrintInstruments()
    