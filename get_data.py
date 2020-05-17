import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import os



def get_data(stocks, start_date, end_date):

    print("downloading data from yahoo finance...")

    for stock in stocks:
        if os.path.exists('./data/data' + stock + '.csv') == True:
            print('./data/data' + stock + '.csv exist, pass\n')
            pass
        else:
            nyyh = web.DataReader(stock, 'yahoo', start_date, end_date)
            nyyh.tail()     
            df = pd.concat([nyyh], axis=1)
            df.dropna(inplace=True)

            df.to_csv('./data/data' + stock + '.csv', encoding='UTF-8')

    # LIBOR
    if os.path.exists("./data/dataLIBOR.csv") == True:
        print("correlated assets exist, pass\n")
        pass
    else:
        nyyh = web.get_data_fred('USDONTD156N',  start_date, end_date)
        df = pd.concat([nyyh], axis=1)
        df.dropna(inplace=True)
        df.to_csv("./data/dataLIBOR.csv", encoding="UTF-8")

        indices = ['000001.SS', '^DJI', '^IXIC', '^GSPC', '^N225', '^HSI', '^FCHI', '^GDAXI']
        savenames = ['SS', 'DJI', 'IXIC', 'GSPC', 'N225', 'HSI', 'FCHI', 'GDAXI']

        # Composite indices
        for indice, savename in zip(indices, savenames):
            nyyh = web.DataReader(indice, 'yahoo',  start_date, end_date)
            df = pd.concat([nyyh], axis=1)
            df.dropna(inplace=True)
            df.to_csv("./data/data" + savename + ".csv", encoding="UTF-8")

        nyyh = web.get_data_fred('DEXJPUS',  start_date, end_date)
        df = pd.concat([nyyh], axis=1)
        df.dropna(inplace=True)
        df.to_csv("./data/dataDEXJPUS.csv", encoding="UTF-8")

        nyyh = web.get_data_fred('DEXCHUS',  start_date, end_date)
        df = pd.concat([nyyh], axis=1)
        df.dropna(inplace=True)
        df.to_csv("./data/dataDEXCHUS.csv", encoding="UTF-8")