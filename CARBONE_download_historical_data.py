import yfinance as yf
import datetime
import csv
import pandas as pd
from typing import Any, Dict

# Tesla
#######
# Create Ticker variables
tsla = yf.Ticker("TSLA")
# Set the time range
tsla_hist = tsla.history(start=datetime.datetime(2021, 2, 24),end=datetime.datetime(2024, 2, 23))
# Export data as csv
tsla_hist.to_csv("Historical_TSLA_data.csv")

# Google
########
goog = yf.Ticker("GOOG")
goog_hist = goog.history(start=datetime.datetime(2021, 2, 24),end=datetime.datetime(2024, 2, 23))
goog_hist.to_csv("Historical_GOOG_data.csv")

# Amazon
########
amzn = yf.Ticker("AMZN")
amzn_hist = amzn.history(start=datetime.datetime(2021, 2, 24),end=datetime.datetime(2024, 2, 23))
amzn_hist.to_csv("Historical_AMZN_data.csv")

