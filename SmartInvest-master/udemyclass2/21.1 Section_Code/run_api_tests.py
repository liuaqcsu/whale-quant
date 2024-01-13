from dateutil.parser import parse
import time
import constants.defs as defs
from infrastructure.collect_data import run_collection

#from api.oanda_api import OandaApi
from infrastructure.instrument_collection import instrumentCollection
#from api.web_options import get_options
from openfx_api.openfx_api import OpenFxApi

if __name__ == "__main__":
    #instrumentCollection.LoadInstrumentsDB()
    api = OpenFxApi()

    #print("\nweb options:")
    #print(get_options())

    #print("\nget_account_summary():")
    #print(api.get_account_summary())

    #instruments = api.get_account_instruments()
    #instrumentCollection.CreateFile(instruments, "./data")
    instrumentCollection.LoadInstruments("./data")
    #print(instrumentCollection.instruments_dict)
    
    
    #print("\nget_candles_df():")
    #print(api.get_candles_df(pair_name="EURUSD", count=-10, granularity="H1"))
    #print(api.get_candles_df(pair_name="EURUSD", count=10, granularity="H1", date_f=parse("2021-01-10T00:00:00")))
    #print(api.last_complete_candle(pair_name="EURUSD", granularity="H1"))

    api.place_trade("EURUSD", 3000, defs.BUY, 1.05900, 1.06200)

    #time.sleep(2)

    #print("Getting open")
    #ot = api.get_open_trades()

    #for t in ot:
    #    print("Got trade:", t)
    #    print("Got trade:", t.id)
    #    time.sleep(2)
    #    print("Closing...")
    #    api.close_trade(t.id)


    #time.sleep(1)
    #print("Getting open")
    #ot = api.get_open_trades()
    #print(ot)
    #print("Done")

    #print(api.get_prices(["EURUSD", "GBPJPY", "GBPUSD"]))
    #print(api.get_pip_value(["EURUSD", "GBPJPY", "GBPUSD"]))

    #run_collection(instrumentCollection, api)





