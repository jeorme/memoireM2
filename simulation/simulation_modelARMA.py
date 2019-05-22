import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt



#global value :
nobs = 10000

#AR simulation and test
arparams = np.array([.75, -.25,.5,-.1])
arparams = np.r_[1, -arparams]
sampleAR = arma_generate_sample(arparams, [1,0,0,0,0], nobs)
fig = plt.plot(sampleAR)
plt.show()
#estimation
ar20 = sm.tsa.ARMA(pd.DataFrame(sampleAR), order=(4,0)).fit(disp=False)
print(ar20.params)
print(ar20.aic, ar20.bic, ar20.hqic)
ar11 = sm.tsa.ARMA(pd.DataFrame(sampleAR), order=(2,2)).fit(disp=False)
print(ar11.params)
print(ar11.aic, ar11.bic, ar11.hqic)

#MA simulation
maparams = np.array([.65, .35])
maparams = np.r_[1, maparams]


