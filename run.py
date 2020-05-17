from . import *
import warnings
warnings.filterwarnings('ignore')
from get_data import *
from correlated_assets import *
from technical_indicator import *
from sentiment import *
from fourier_transforms import *
from ARIMA import *
from feature_importance import *
from autoencoder import *
from PCA import *
from process_data import *
from GAN import *
import os

stocks = ['GS', 'JPM', 'MS']

start_date = dt.datetime(2012,9,4)
end_date = dt.datetime(2016,10,31)

codes = open('stock-code')
code = codes.readline()

result = open('./data/result/final-result.txt', 'w+')
result.close()

while code:
    code = code.strip('\s')
    code = code.strip('\n')
    print("processing "+code+"...\n")
    '''
    if os.path.exists('./data/result/'+code+'.png') == True:
        print("already processed, pass")
        code = codes.readline()
        continue
    '''
    if os.path.exists('./data/train/train_data'+code+'.npy') == False:
        try:
            get_data([code,'JPM','MS'], start_date, end_date)
        except Exception as e:
            print('code not exsit\n')
            code = codes.readline()
            continue
        correlated_assets(code)
        get_dataset(code)
        get_sentiment(code)
        get_FFT(code)
        get_ARIMA(code)
        get_FI(code)
        get_autoencoder(code)
        get_PCA(code)
        process_data(code)
    else:
        print("training data already exist\n")
    opt = {"lr":0.001,"epoch":100}
    loss = train(code, opt, plot=False, verbose=True)
    # loss = tunr(code)
    result = open('./data/result/final-result.txt', 'a')
    result.write(code + "," + str(loss) + "\n")
    result.close()
    code = codes.readline()
codes.close()

