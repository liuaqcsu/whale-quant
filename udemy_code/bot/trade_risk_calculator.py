from openfx_api.openfx_api import OpenFxApi
from infrastructure.instrument_collection import instrumentCollection as ic

BASE = 10000
MINUMUM = 1000

def get_trade_size(api: OpenFxApi, pair, loss, trade_risk, log_message):

    pip_values = api.get_pip_value([pair])    

    if pip_values is None or len(pip_values.keys()) == 0:
        log_message("get_trade_size() pip_values is none", pair)
        return 0

    our_pip_value = pip_values[pair]    
        
    log_message(f"get_trade_size() our_pip_value {our_pip_value:.6f}", pair)

    our_instrument = ic.instruments_dict[pair]


    pipLocation = our_instrument.pipLocation
    num_pips = loss / pipLocation
    per_pip_loss = trade_risk / num_pips


    ratio = per_pip_loss / our_pip_value


    trade_pure = BASE * ratio

    trade_size = int(trade_pure / our_instrument.TradeAmountStep) * our_instrument.TradeAmountStep
        
    log_message(f"get_trade_size() num_pips:{num_pips} per_pip_loss:{per_pip_loss} ratio:{ratio} trade_pure:{trade_pure} trade_size:{trade_size}", pair)
    
    if trade_size < MINUMUM:
        return 0
    
    return trade_size

    

