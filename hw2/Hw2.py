import numpy as np
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt


def generateAR3Points(params):
    ''' use lib 
    np.random.seed(12345)
    arparams = np.array(params)
    maparams = np.array([0.])
    ar = np.r_[1, -arparams] # add zero-lag and negate
    ma = np.r_[1, maparams] # add zero-lag
    ar3 = smt.arma_generate_sample(ar, ma, 500)
    '''
    np.random.seed(1)
    n_samples = int(500)
    
    x = w = np.random.normal(size=n_samples)

    for t in range(3,n_samples):
        x[t] = params[0] + params[1] * x[t-1] + params[2] * x[t-2] + params[3] * x[t-3] + w[t]
    return x

def drawACF(points):
    #plt.subplot(2,1,1)
    plot_acf(points,lags = 30,markersize = 3 )
    plt.title('ACF of AR(3)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
    
def drawPACF(points):
    #plt.subplot(2,1,2)
    plot_pacf(points, lags=30,markersize = 3 )
    plt.title('PACF of AR(3)')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()

def SelectOrder(points):
    order = smt.AR(points).select_order(maxlag=10, ic='aic', trend='nc')
    return order

def getParameters(points):
    mdl = smt.AR(points).fit(maxlag=10, ic='aic', trend='nc')
    return mdl

def checkAccuracy(points,mdl):
    drawACF(mdl.resid)

def drawPlot(points,mdl):
    forecastP = mdl.predict(start=4, end=len(points))
    x = range(1,501) 
    plt.plot(x, points,color = 'b',linewidth=1,alpha=0.6,label='observation')
    x = range(4,501)
    plt.plot(x, forecastP,color = 'r',linewidth=1,alpha=0.6,label='forecast')
    plt.legend(loc='upper right')
    plt.ylim((-4,4))
    plt.show()

#Q1
param = [0.01,0.1,0.0,-0.1]
ar3 = generateAR3Points(param)
#Q2
drawACF(ar3)
#Q3
drawPACF(ar3)
#Q4
order = SelectOrder(ar3)
print(str(order))
#Q5
mdl = getParameters(ar3)
print(mdl.params)
#Q6
checkAccuracy(ar3,mdl)

drawPlot(ar3,mdl)