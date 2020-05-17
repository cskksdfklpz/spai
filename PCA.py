from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle 
import numpy as np
import pandas as pd

def get_PCA(stock):

    dataset_total_df = pd.read_csv("./data/features"+stock+".csv")
    if "PCA1" in list(dataset_total_df):
        pass
    else:

        print("computing PCA...")
        f = open("./data/VAE/"+stock, "rb")
        vae_added_df = pickle.load(f)
        pca = PCA(n_components=0.9)
        x_pca = StandardScaler().fit_transform(vae_added_df.asnumpy())
        principalComponents = pca.fit_transform(x_pca)

        print("==pca.n_components_==")
        print(pca.n_components_)
        print(principalComponents)
        print(pca.explained_variance_ratio_)
        header = []
        for i in range(1, pca.n_components_+1):
            header.append("PCA"+str(i))
        df_PCA = pd.DataFrame(principalComponents, index=dataset_total_df["Date"], columns=header)
        dataset_total_df = pd.merge(dataset_total_df, df_PCA, on="Date", how="left")
        dataset_total_df = dataset_total_df.drop(['Unnamed: 0'],axis=1) 
        dataset_total_df.to_csv("./data/features"+stock+".csv")

