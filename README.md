# Yahoo-TimeSeries-Forecast
For this project I used python to download 3 years worth of stock data from Tesla, Amazon and Google and export them as csv files (Seen in CARBONE_download_historical_data.py).
I also wrote a python scrypt to display and anlyze the movement of the stock prices (Seen in CARBONE_Time_Series_Model).
I used the sktime python library to develop a simple naive forecaster model using a mean strategy with an annual seasonality parameter.
A deeper explanation of the model, my process and final predicitons can be seen in CARBONE_TimeSeries_Expl.docx.
