import pandas as pd
import numpy as np
import datetime as dt
import warnings
import matplotlib.pyplot as plt
from collections import deque 
warnings.filterwarnings("ignore")  


def parser(x):
    return dt.datetime.strptime(x,'%Y-%m-%d')

def get_FFT(stock):

    dataset_ex_df = pd.read_csv('./data/data'+stock+'.csv', header=0, parse_dates=[0], date_parser=parser)
    data_FT = dataset_ex_df[['Date', 'Close']]    


    dataset_total_df = pd.read_csv("./data/features"+stock+".csv", date_parser=parser)
    if 'Fourier3' in list(dataset_total_df):
        pass
    else:
        print("computing fourier transforms...")
        close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))  
        fft_df = pd.DataFrame({'fft':close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))  
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))   
        plt.figure(figsize=(14, 7), dpi=400)
        fft_list = np.asarray(fft_df['fft'].tolist())
        for num_ in [3, 6, 9, 100]:
            fft_list_m10 = np.copy(fft_list)
            fft_list_m10[num_:-num_] = 0
            plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
        
        plt.plot(data_FT['Close'], label='Real')
        plt.xlabel('Days')
        plt.ylabel('USD')
        plt.title('Figure 3: '+stock+' (close) stock prices & Fourier transforms')
        plt.legend()
        plt.savefig('./data/FFT/'+stock+'.png', format='png', dpi=400)

        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[3:-3] = 0
        fft_df3 = pd.DataFrame({'Fourier3': np.abs(np.fft.ifft(fft_list_m10))})

        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[6:-6] = 0
        fft_df6 = pd.DataFrame({'Fourier6': np.abs(np.fft.ifft(fft_list_m10))})

        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[9:-9] = 0
        fft_df9 = pd.DataFrame({'Fourier9': np.abs(np.fft.ifft(fft_list_m10))})

        data_FT['Fourier3'] = fft_df3
        data_FT['Fourier6'] = fft_df6
        data_FT['Fourier9'] = fft_df9

        data_FT["Date"] = pd.to_datetime(data_FT["Date"])
        dataset_total_df["Date"] = pd.to_datetime(dataset_total_df["Date"])
        
        dataset_total_df = pd.merge(dataset_total_df, data_FT, on="Date", how="left") 
        dataset_total_df = dataset_total_df.drop(['Unnamed: 0'],axis=1)   
        dataset_total_df.to_csv("./data/features"+stock+".csv")

