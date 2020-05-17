import pandas as pd 
import numpy as np 

def process_data(stock):
    print("processing training data...")
    df = pd.read_csv("./data/features"+stock+".csv")
    df_ = df.drop(columns=["Date"], axis = 1)
    np_features = df_.to_numpy()
    np.save("./data/train/train_data"+stock+".npy", np_features)