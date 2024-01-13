import datetime as dt
import pytz


class ApiPrice():

    def __init__(self, api_ob):
        self.instrument = api_ob['Symbol']
        self.ask = api_ob['BestBid']['Price']
        self.bid = api_ob['BestAsk']['Price']
        self.time = dt.datetime.fromtimestamp(api_ob['Timestamp']/1000, pytz.timezone("UTC")) 

    
    def __repr__(self):
        return f"ApiPrice() {self.instrument} {self.ask} {self.bid} {self.time}"


    def get_dict(self):
        return dict(
            instrument=self.instrument,
            time = self.time,
            ask=self.ask,
            bid=self.bid
        )
