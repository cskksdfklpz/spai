from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
import datetime as dt
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")  

def parser(x):
    return dt.datetime.strptime(x,'%Y-%m-%d')

def get_ARIMA(stock):

    dataset_ex_df = pd.read_csv('./data/data'+stock+'.csv', header=0, parse_dates=[0], date_parser=parser)
    data_FT = dataset_ex_df[['Date', 'Close']]
    series = data_FT['Close']     

    dataset_total_df = pd.read_csv("./data/features"+stock+".csv", date_parser=parser)
    if 'ARIMA' in list(dataset_total_df):
        pass
    else:
        print("computing ARIMA prediction")
        model = ARIMA(series, order=(5, 1, 0)) 
        model_fit = model.fit(disp=0)          
        print(model_fit.summary())

        X = series.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]   
        history = [x for x in train]             

        predictions = list()                      
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)                  
            
        error = mean_squared_error(test, predictions)  #
        print('Test MSE: %.3f' % error)

        train = pd.DataFrame(train)
        predictions = pd.DataFrame(predictions)
        ARIMA_predictions = pd.concat([train,predictions])
        ARIMA_predictions.index = range(len(X))
        ARIMA_predictions.columns = ['ARIMA']
        data_FT['ARIMA'] = ARIMA_predictions
        data_FT["Date"] = pd.to_datetime(data_FT["Date"])
        dataset_total_df["Date"] = pd.to_datetime(dataset_total_df["Date"])
        dataset_total_df = pd.merge(dataset_total_df, data_FT)  
        dataset_total_df = dataset_total_df.drop(['Unnamed: 0'],axis=1) 
        dataset_total_df.to_csv("./data/features"+stock+".csv")