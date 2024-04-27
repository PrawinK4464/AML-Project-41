import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate some sample time series data
date_range = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
data = np.random.randn(len(date_range))
ts = pd.Series(data, index=date_range)

# Plot the time series data
ts.plot(figsize=(10, 6))
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Decompose the time series into trend, seasonal, and residual components
decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Perform a simple time series forecasting (e.g., using ARIMA model)
model = sm.tsa.ARIMA(ts, order=(1, 1, 1))  # Example ARIMA model with (p, d, q) = (1, 1, 1)
results = model.fit()
forecast = results.forecast(steps=12)  # Forecasting 12 steps ahead

# Plot the original data and the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(ts, label='Original')
plt.plot(forecast, color='red', label='Forecast')
plt.title('Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.show()
