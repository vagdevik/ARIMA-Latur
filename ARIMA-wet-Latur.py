import numpy as np
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

data = pd.read_csv("data.csv",sep="\t")
data.head()

months = data.columns
years = data.Year
Time = []
for year in years:
    for month in months:
        if(month!='Year'):
            Time.append(str(year)+" "+month)
Time = pd.Index(Time)            
# print Time
# print Time.shape

Values = []
for index, row in data.iterrows():
    r = []
    r = list(row)
    r.pop(0)
    Values.extend(r)

Values = pd.Index(Values)
# print Values
# print Values.shape

series = pd.DataFrame({'Time': Time, 'Values': Values})
dummy = series
dummy.head()

series.index = pd.to_datetime(series.Time)
series.rename(columns={"Time":"Date"})
# series.drop(["Date"],axis=1)

series_data = series.Values
minimum=series_data.min()
maximum=series_data.max()

series_data.plot()
axes = pyplot.gca()
axes.set_ylim([minimum,maximum]) #setes ticks on y-axis according to min and max 

pyplot.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

pyplot.title("Data Plot")
pyplot.xlabel("Time")
pyplot.ylabel("Values")

pyplot.show()

# checking the mean and variance for stationarity
X = series.Values
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

yearly_mean = series_data.rolling(window=12).mean()
yearly_std = series_data.rolling(window=12).std()

orig = pyplot.plot(series.Values,color='blue',label='Original')
mean = pyplot.plot(yearly_mean,color='red',label='Mean')
std = pyplot.plot(yearly_std,color='black',label='Standard Deviation')

pyplot.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

pyplot.title("Data Plot")
pyplot.xlabel("Time")
pyplot.ylabel("Values")


pyplot.legend(loc='best')

#dickey-fuller test
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(series_data)    

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
kpss_test(series_data)

#extract data values
zt = np.array(series_data)

#mean of data
mean = np.mean(zt)

#variance of data
#var = zt.var()
c0 = np.sum((zt - mean)*(zt - mean))/len(zt)

#calculate lag-wise auto-correlation
corr_coeffs=[]
lags=[]
for k in range(len(zt)):
    series_1 = zt[k:]
    series_2 = zt[:len(zt)-k]
    
    if len(series_1)!=len(series_2):
        print "Error!!!"
    else:
        num = np.sum((series_1 - mean)*(series_2 - mean))
        den = c0*len(zt)
        coeff = num/den
        
        corr_coeffs.append(coeff)
        lags.append(k)

#print the corr_coeffs and lags
# print "************************"
# print "correlation coeffs: ",corr_coeffs
# print "------------"
# print "lags: ",lags
# print "************************"

#plot autocorr vs lag
pyplot.title("calculated")
pyplot.bar(lags,corr_coeffs,width=0.2)
pyplot.axhline(0)
pyplot.show()

#plot the same using in-built func
plot_acf(zt,lags=20)
pyplot.show()

#plot pacf using in-built function
plot_pacf(zt,lags=20)
pyplot.show()	

lag_acf = acf(series_data,nlags=20)
lag_pacf = pacf(series_data,nlags=20,method='ols')

pyplot.subplot(121)
pyplot.plot(lag_acf)
pyplot.axhline(y=0,linestyle='--',color='gray')
pyplot.axhline(y=-1.96/np.sqrt(len(series_data)),linestyle='--',color='gray')
pyplot.axhline(y=1.96/np.sqrt(len(series_data)),linestyle='--',color='gray')
pyplot.title("ACF")

pyplot.subplot(122)
pyplot.plot(lag_pacf)
pyplot.axhline(y=0,linestyle='--',color='gray')
pyplot.axhline(y=-1.96/np.sqrt(len(series_data)),linestyle='--',color='gray')
pyplot.axhline(y=1.96/np.sqrt(len(series_data)),linestyle='--',color='gray')
pyplot.title("PACF")

# AR
model = ARIMA(series_data,order=(2,0,0))
results_AR = model.fit(disp = -1)
pyplot.plot(series_data)
pyplot.plot(results_AR.fittedvalues,color='red')
pyplot.title('RSS: %.4f'% sum(results_AR.fittedvalues-series_data)**2)


# MA
model = ARIMA(series_data,order=(0,0,4))
results_MA = model.fit(disp = -1)
pyplot.plot(series_data)
pyplot.plot(results_MA.fittedvalues,color='red')
pyplot.title('RSS: %.4f'% sum(results_MA.fittedvalues-series_data)**2)


model = ARIMA(series_data,order=(3,0,2))
# model = ARIMA(series_data,order=(2,0,4))
results_ARIMA = model.fit(disp = -1)
pyplot.plot(series_data)
pyplot.plot(results_ARIMA.fittedvalues,color='red')
pyplot.title('RSS: %.4f'% sum(results_ARIMA.fittedvalues-series_data)**2)


predictions_AR = pd.Series(results_AR.fittedvalues,copy=True)
predictions_AR.head()

predictions_MA = pd.Series(results_MA.fittedvalues,copy=True)
predictions_MA.head()

predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues,copy=True)
predictions_ARIMA.head()

# convert to cummulative values
predictions_AR_cumsum = predictions_AR.cumsum()
predictions_AR_cumsum.head()

# convert to cummulative values
predictions_MA_cumsum = predictions_MA.cumsum()
predictions_MA_cumsum.head()

# convert to cummulative values
predictions_ARIMA_cumsum = predictions_ARIMA.cumsum()
predictions_ARIMA_cumsum.head()

# convert to cummulative values
# predictions_ARIMA_log = pd.Series(series.)
predictions_AR_log = pd.Series(series_data, index=dummy.index)
predictions_AR_e = np.exp(predictions_AR_log)

# convert to cummulative values
# predictions_ARIMA_log = pd.Series(series.)
predictions_AR_log = pd.Series(series_data, index=dummy.index)
predictions_AR_e = np.exp(predictions_AR_log)

predictions_MA_log = pd.Series(series_data, index=dummy.index)
predictions_MA_e = np.exp(predictions_MA_log)

predictions_ARIMA_log = pd.Series(series_data, index=dummy.index)
predictions_ARIMA_e = np.exp(predictions_ARIMA_log)

results_AR.plot_predict(1,492)
# results_AR.forecast(steps=120)

results_MA.plot_predict(1,492)
# results_MA.forecast(steps=120) #for pvalues

results_ARIMA.plot_predict(1,492)



