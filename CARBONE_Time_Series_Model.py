#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import sktime
import matplotlib.pyplot as plt
import numpy as np
from sktime.forecasting.naive import NaiveForecaster


# In[66]:


# Tesla Forecast
################
tsla = pd.read_csv('Historical_TSLA_data.csv')
tsla = tsla['Close']
print(tsla)


# In[67]:


# Plotting time series data
plt.figure(figsize=(10, 6))
plt.plot(tsla)
plt.title("Tesla Stock Prices Over Time (Daily)")
plt.xlabel("Time")
plt.ylabel("Closing Stock Price")
plt.show()


# In[68]:


# Train Test Split
# Split between training and test sets
train = tsla.iloc[:550]
test = tsla.iloc[550:]


# In[69]:


# Plottig training and testing data with eachother
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.title('Train/Test Split of Tesla')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[70]:


# Creating model and making predictions
forecaster = NaiveForecaster(strategy = "mean", sp=260) # last, mean, drift
forecaster.fit(train)

FH = np.arange(1,len(test) + 1)
pred = forecaster.predict(fh = FH)
pred


# In[71]:


# Plotting training, testing and prediction data with eachother
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.plot(pred.index, pred.values, label='Prediction', color='green')
plt.title('Train/Test Split with Prediction of Tesla')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[72]:


# Amazon Forecast
#################
amzn = pd.read_csv('Historical_AMZN_data.csv')
amzn = amzn['Close']
print(amzn)


# In[73]:


# Plotting time series data
plt.figure(figsize=(10, 6))
plt.plot(amzn)
plt.title("Amazon Stock Prices Over Time (Daily)")
plt.xlabel("Time")
plt.ylabel("Closing Stock Price")
plt.show()


# In[74]:


# Train Test split
# Split between training and test sets
train = amzn.iloc[:550]
test = amzn.iloc[550:]


# In[75]:


# Plotting training and testing data
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.title('Train/Test Split of Amazon')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[76]:


# Creating model and making predictions
forecaster = NaiveForecaster(strategy = "mean", sp=260) # last, mean, drift
forecaster.fit(train)

FH = np.arange(1,len(test) + 1)
pred = forecaster.predict(fh = FH)
pred


# In[77]:


# Plotting training, testing and prediciton data with eachother
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.plot(pred.index, pred.values, label='Prediction', color='green')
plt.title('Train/Test Split with Prediction of Amazon')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[78]:


# Google Forecast
#################
goog = pd.read_csv('Historical_GOOG_data.csv')
goog = goog['Close']
print(goog)


# In[79]:


# Plotting time series data
plt.figure(figsize=(10, 6))
plt.plot(goog)
plt.title("Google Stock Prices Over Time (Daily)")
plt.xlabel("Time")
plt.ylabel("Closing Stock Price")
plt.show()


# In[80]:


# Train Test split
# Split between training and test sets
train = goog.iloc[:550]
test = goog.iloc[550:]


# In[81]:


# Plotting training and testing data with eachother
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.title('Train/Test Split of Google')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[82]:


# Creating model and making predictions
forecaster = NaiveForecaster(strategy = "mean", sp = 260) # last, mean, drift
forecaster.fit(train)

FH = np.arange(1,len(test) + 1)
pred = forecaster.predict(fh = FH)
pred


# In[83]:


# Plotting training, testing and prediction data with eachother
plt.figure(figsize=(10, 6))
plt.plot(train.index, train.values, label='Train', color='blue')
plt.plot(test.index, test.values, label='Test', color='red')
plt.plot(pred.index, pred.values, label='Prediction', color='green')
plt.title('Train/Test Split with Prediction of Google')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()


# In[ ]:




