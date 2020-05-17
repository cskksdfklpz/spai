import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")   

def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    print("computing MA-7...")
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean().fillna(method='backfill')
    print("computing MA-21...")
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean().fillna(method='backfill')
     
    # Create MACD
    print("computing EMA-26...")
    dataset['26ema'] = pd.DataFrame.ewm(dataset['Close'], span=26).mean().fillna(method='backfill')
    print("computing EMA-12...")
    dataset['12ema'] = pd.DataFrame.ewm(dataset['Close'], span=12).mean().fillna(method='backfill')
    print("computing MACD...")
    dataset['MACD'] = (dataset['12ema']-dataset['26ema']).fillna(method='backfill')
    
    # Create Bollinger Bands
    print("computing 20sd...")
    dataset['20sd'] = dataset['Close'].rolling(20).std().fillna(method='backfill')
    print("computing upper band...")
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2).fillna(method='backfill')
    print("computing lower band...")
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2).fillna(method='backfill')
    
    # Create Exponential moving average
    print("computing EMA...")
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean().fillna(method='backfill')
     
    # Create Momentum
    print("computing momentum...")
    dataset['momentum'] = dataset['Close']-1
     
    #Create log_momentum
    print("computing log momentum...")
    dataset['log_momentum'] = dataset['momentum'].apply(lambda x:math.log(max(x,1))).fillna(method='backfill')
     
    return dataset



# plot the indicators
def plot_technical_indicators(dataset, stock, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
     
    dataset = dataset.iloc[-last_days:, :]  
    x_ = range(3, dataset.shape[0])
    x_ = list(dataset.index)
     
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['Close'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for '+stock+' - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')
    plt.legend()
    plt.savefig('./data/TF/'+stock+'.png', format='png', dpi=400)
     
def get_dataset(stock):
    dataset_total_df = pd.read_csv("./data/features"+stock+".csv")
    if 'ma7' in list(dataset_total_df):
        pass
    else:
        dataset_TI_df = get_technical_indicators(dataset_total_df[['Date','Close']]) 
        dataset_total_df = pd.merge(dataset_total_df, dataset_TI_df)
        dataset_total_df = dataset_total_df.drop(['Unnamed: 0'],axis=1) 
        dataset_total_df.to_csv("./data/features"+stock+".csv")
        plot_technical_indicators(dataset_total_df, stock, 400)
