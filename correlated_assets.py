import warnings
import mxnet as mx
import pandas as pd
import datetime as dt

warnings.filterwarnings("ignore")        

context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)

def parser(x):
    return dt.datetime.strptime(x,'%Y-%m-%d')

def correlated_assets(stock):

    print("loading "+stock+ " stock price...")
    dataset_ex_df = pd.read_csv('./data/data' + stock + '.csv', header=0, parse_dates=[0], date_parser=parser)
    print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0])) 
    dataset_total_df = dataset_ex_df.copy()

    print("loading JPM stock price...")
    JPM = pd.read_csv('./data/dataJPM.csv', header=0)
    dataset_total_df['JPM'] = JPM[['Close']]             

    print("loading MS stock price...")
    MS = pd.read_csv('./data/dataMS.csv', header=0)
    dataset_total_df['MS'] = MS[['Close']]  

    print("loading LIBOR...")
    LIBOR = pd.read_csv('./data/dataLIBOR.csv', header=0, parse_dates=[0], date_parser=parser)
    LIBOR.columns = ['Date','LIBOR']
    dataset_total_df = pd.merge(dataset_total_df, LIBOR)  

    print("loading VIX...")
    vix = pd.read_csv('./data/vixcurrent.csv', header=0)
    vix_close = vix.loc[2184:3230, ['VIX Close']]
    vix_close.index = range(1047)
    dataset_total_df['VIX'] = vix_close  

    print("loading stock index...")
    savenames = ['SS', 'DJI', 'IXIC', 'GSPC', 'N225', 'HSI', 'FCHI', 'GDAXI']

    for savename in savenames:
        indice = pd.read_csv('./data/data' + savename + '.csv', header=0, parse_dates=[0], date_parser=parser)
        indice = indice[['Date','Close']]
        indice.columns = ['Date', savename + ' Close']
        dataset_total_df = pd.merge(dataset_total_df, indice)

    print("loading currency...")
    DEXJPUS = pd.read_csv('./data/dataDEXJPUS.csv', header=0, parse_dates=[0], date_parser=parser)
    DEXJPUS.columns = ['Date','DEXJPUS']
    dataset_total_df = pd.merge(dataset_total_df, DEXJPUS)  

    DEXCHUS = pd.read_csv('./data/dataDEXCHUS.csv', header=0, parse_dates=[0], date_parser=parser)
    DEXCHUS.columns = ['Date','DEXCHUS']
    dataset_total_df = pd.merge(dataset_total_df, DEXCHUS)

    dataset_total_df.to_csv("./data/features"+stock+".csv")