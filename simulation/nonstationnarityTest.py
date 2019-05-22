import statsmodels.tsa.arima_process as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
ar=[1,0.7,0.6,.5,.4]
ma=[1]
n=1000
sample = ts.arma_generate_sample(ar, ma, n)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(311)
ax.plot(sample)
ax1 = fig.add_subplot(312)
fig = sm.graphics.tsa.plot_acf(sample, lags=40, ax=ax1)
ax2 = fig.add_subplot(313)
fig = sm.graphics.tsa.plot_pacf(sample, lags=40, ax=ax2)
plt.show()
result = adfuller(sample)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

ar=[1,0,-1.5,0,0.5]
ma=[1]

sample = ts.arma_generate_sample(ar, ma, n)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(311)
ax.plot(sample)
ax1 = fig.add_subplot(312)
fig = sm.graphics.tsa.plot_acf(sample, lags=40, ax=ax1)
ax2 = fig.add_subplot(313)
fig = sm.graphics.tsa.plot_pacf(sample, lags=40, ax=ax2)
plt.show()
result = adfuller(sample)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))