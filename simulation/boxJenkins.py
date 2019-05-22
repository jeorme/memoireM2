import statsmodels.tsa.arima_process as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt

def boxJenkinsAlgo(sample):
    d = getD(sample)
    acf  = sm.tsa.stattools.acf(sample,qstat =True, alpha=0.01)
    pacf = sm.tsa.stattools.pacf(sample,alpha  =0.01)
    p = getP(acf[3],0.01)
    q = getQ(pacf[1])
    aic = []
    Best = None
    for i in range(d+1):
        for j in range(p+1):
            for k in range(q+1):
                result = sm.tsa.statespace.SARIMAX(sample,order=(j,i,k)).fit(disp=False)
                aic.append(result.aic)
                if result.aic<= np.min(aic):
                    Best = result

    plt.plot(aic)
    plt.show()
    print(aic)
    print(Best.summary())
    return Best



def getD(sample):
    result = adfuller(sample)
    if result[1]<0.05:
        return 0
    else:
        return 1

def getP(acf,alpha):
   if( max(acf)<alpha):
       return 0
   index = np.where(acf.values>alpha)
   return min(index) - 1


def getQ(vals):
    index= 0
    for val in vals:
        if 0 >=val[0] and 0<=val[1]:
            return index - 1
        index +=1
    return 0


ar=[1,0.8,0.5,0.6,0.4]
ma=[1]

sample = ts.arma_generate_sample(ar, ma, 1000)
boxJenkinsAlgo(sample)