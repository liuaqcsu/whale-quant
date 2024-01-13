from simulation.ema_macd import run_ema_macd
from simulation.ma_cross import run_ma_sim

from infrastructure.instrument_collection import instrumentCollection

instrumentCollection.LoadInstruments("./data")

#run_ma_sim()
run_ema_macd(instrumentCollection)