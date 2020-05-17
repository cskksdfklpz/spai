import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")  

def parser(x):
    return dt.datetime.strptime(x,'%Y-%m-%d')

def plot_sentiment(dataset, stock, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    dataset = dataset.iloc[-last_days:, :]  
    plt.plot(dataset['sentiment'],label='sentiment score', color='g',linestyle='-',marker='o')
    plt.title('Sentiment score for '+stock+' - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()
    plt.savefig('./data/sentiment/fig/'+stock+'.png', format='png', dpi=400)

def get_sentiment(stock):

    dataset_total_df = pd.read_csv("./data/features"+stock+".csv")
    if 'sentiment' in list(dataset_total_df):
        pass
    else:
        # see finBERT for the algorithm to compute sentiment score
        print("loading sentiment score...")
        dataset_senti_df = pd.read_csv('./data/sentiment/'+stock+'.csv')
        dataset_senti_df.columns = ['Date', 'sentiment']
        # dataset_senti_df["Date"] = pd.to_datetime(dataset_senti_df["Date"])
        # dataset_total_df["Date"] = pd.to_datetime(dataset_total_df["Date"])
        dataset_total_df = pd.merge(dataset_total_df, dataset_senti_df, on="Date", how="left")
        dataset_total_df["sentiment"] = dataset_total_df["sentiment"].fillna(0)
        dataset_total_df = dataset_total_df.drop(['Unnamed: 0'],axis=1) 
        dataset_total_df.to_csv("./data/features"+stock+".csv")
    plot_sentiment(dataset_total_df, stock , 800)

