#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

get_ipython().system('pip install pmdarima --quiet')
import pmdarima as pm

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[79]:


df = pd.read_csv('monthly-milk-production-pounds-p.csv', parse_dates = ['Month'], index_col = 'Month')
df.head()


# In[80]:


df.tail()


# In[81]:


df.shape


# In[82]:


#168 monthly milk production records are present from Jan 1962 - Dec 1975


# In[83]:


df.describe()


# In[84]:


df.isna().sum()


# In[85]:


#No null values are present.


# In[86]:


df.plot(figsize = (10,5))
plt.title('Monthly Milk Production')
plt.show()


# In[87]:


#We can observe that there is an increasing trend and very strong seasonality in our data.


# In[88]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))
df.hist(ax = ax1)
df.plot(kind = 'kde', ax = ax2)
plt.show()


# In[89]:


# Remove missing values
df_clean = df.dropna()

# or Impute missing values
# df_clean = df.fillna(method='ffill')  # Forward fill missing values
# df_clean = df.fillna(method='bfill')  # Backward fill missing values
# df_clean = df.fillna(df.mean())  # Fill missing values with mean
# df_clean = df.fillna(df.median())  # Fill missing values with median
# Or any other imputation method suitable for your data

# Perform seasonal decomposition on the cleaned data
decomposition = seasonal_decompose(df_clean['production'], period=12, model='additive')
plt.rcParams['figure.figsize'] = 10, 5
decomposition.plot()


# In[90]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(10, 5))

# Plot ACF
plot_acf(df['production'], lags=40, ax=ax1)

# Plot PACF
plot_pacf(df['production'], lags=40, ax=ax2)

plt.subplots_adjust(hspace=0.5)
plt.show()


# In[91]:


#ADF Test


# In[92]:


from statsmodels.tsa.stattools import adfuller

def adfuller_test(production):
    # Remove missing and infinite values
    production_clean = production.replace([np.inf, -np.inf], np.nan).dropna()
    result = adfuller(production_clean)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', '#Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis (Ho), reject the null hypothesis. Data has no unit root and is stationary.')
    else:
        print('Weak evidence against the null hypothesis, hence ACCEPT Ho. and the series is Not Stationary.')

adfuller_test(df['production'])


# In[93]:


df1 = df.diff().diff(12).dropna()


# In[94]:


adfuller_test(df1['production'])


# In[95]:


df1.plot(figsize=(10,5))
plt.title('Monthly Milk Production')
plt.show()


# In[96]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))

ax1 = autocorrelation_plot(df['production'], ax = ax1)
ax1.set_title('Non - Stationary Data')

ax2 = autocorrelation_plot(df1['production'], ax = ax2)
ax2.set_title('Stationary Data')

plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[97]:


fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, figsize = (10,5))

ax1 = plot_acf(df1['production'], lags = 40, ax = ax1)
ax2 = plot_pacf(df1['production'], lags = 40, ax = ax2)

plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[98]:


## Model Parameter Estimation


# In[99]:


import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# Remove NaN values
df_clean = df.dropna()

# or Impute NaN values
# df_clean = df.fillna(method='ffill')  # Forward fill NaN values
# df_clean = df.fillna(method='bfill')  # Backward fill NaN values
# df_clean = df.fillna(df.mean())  # Fill NaN values with mean
# df_clean = df.fillna(df.median())  # Fill NaN values with median
# Or any other imputation method suitable for your data

# Fit ARIMA model
model = pm.auto_arima(df_clean['production'], d=1, D=1,
                      seasonal=True, m=12, 
                      start_p=0, start_q=0, max_order=6, test='adf', trace=True)


# In[100]:


model.summary()


# In[101]:


train = df[:int(0.85*(len(df)))]
test = df[int(0.85*(len(df))):]

train.shape, test.shape


# In[102]:


model = SARIMAX(train['production'],
                order = (1,1,0), seasonal_order = (0,1,1,12))
results = model.fit(disp = False)
results.summary()


# In[103]:


##### We have created a SARIMAX model using the best parameters on our training data giving us an AIC Score of 897.205


# In[104]:


## Model Validation


# In[105]:


results.plot_diagnostics(figsize = (15, 5))
plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[106]:


start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start = start, end = end, dynamic = False, typ = 'levels').rename('SARIMA(1,1,0)(0,1,1,12) Test predictions')


# In[107]:


for i in range(len(predictions)):
    print(f"predicted = {predictions[i]:<11.10}, expected = {test['production'][i]}")


# In[108]:


import matplotlib.pyplot as plt

# Assuming 'test' contains the observed production data and 'predictions' contains the predicted values

title = 'Monthly Milk production'
ax = test['production'].plot(legend=True, figsize=(9, 4), title=title)  # Plot observed production data
predictions.plot(legend=True, ax=ax)  # Plot predicted values on the same plot
ax.autoscale(axis='x', tight=True)  # Autoscale x-axis

plt.show()


# In[109]:


import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Assuming 'test' contains the observed production data and 'predictions' contains the predicted values

# Drop rows with NaN values from both test and predictions
test_clean = test.dropna()
predictions_clean = predictions.dropna()

# Ensure lengths match by truncating the longer series
min_length = min(len(test_clean['production']), len(predictions_clean))
test_clean = test_clean.iloc[:min_length]
predictions_clean = predictions_clean.iloc[:min_length]

# Calculate evaluation metrics
evaluation_results = pd.DataFrame({'r2_score': [r2_score(test_clean['production'], predictions_clean)]}, index=[0])
evaluation_results['mean_absolute_error'] = mean_absolute_error(test_clean['production'], predictions_clean)
evaluation_results['mean_squared_error'] = mean_squared_error(test_clean['production'], predictions_clean)

print(evaluation_results)


# In[110]:


forecast = results.get_prediction(start = '1975-12-01', end = '1980-12-01')
idx = np.arange(len(forecast.predicted_mean))

forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

fig, ax = plt.subplots()
df.plot(ax = ax, label='observed')
forecast_values.plot(ax = ax, label = 'predicted', alpha = 0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
plt.legend()
ax.set_title('Forecast of production')
plt.show()


# In[111]:


##Here is forecast for the next four years


# In[ ]:




