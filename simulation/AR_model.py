import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt


arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]
nobs = 2500
sample = arma_generate_sample(arparams, maparams, nobs)
y=pd.DataFrame(sample)
arma_mod = sm.tsa.ARMA(y, order=(2,2))
arma_res = arma_mod.fit(trend='nc', disp=1)
arma_res.summary()
arparams2 = np.array([.15, -.05])
maparams2 = np.array([.95, 1.5])
arparams2 = np.r_[1, -arparams2]
maparams2 = np.r_[1, maparams2]
nobs = 2500
sample2 = arma_generate_sample(arparams2, maparams2, nobs)
y2= pd.DataFrame(np.r_[sample,sample2])
arma_mod2 = sm.tsa.ARMA(y2, order=(2,2))
arma_res2 = arma_mod2.fit(trend='nc', disp=1)
fig = plt.plot(sample2)
plt.show()
print(arma_res2.summary())

