import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")  

def train(stock, opt, plot = False, verbose = False):

    data = np.load('./data/train/train_data'+stock+'.npy')
    n_F = data.shape[1]
    pos_target = 3

    class Seq2Seq(nn.Module):                                    #双层
        def __init__(self):
            super(Seq2Seq, self).__init__()
            self.embedding1 = nn.Linear(n_F,64)
            self.embedding2 = nn.Linear(64,64)
            self.encoder = nn.GRU(64,64, num_layers = 2)
            self.decoder = nn.GRU(64,64, num_layers = 2)
            self.mid_con = nn.Linear(64,64)
            self.to_v = nn.Linear(64,1)
            self.drop = nn.Dropout(0.2)
            self.softsign = torch.nn.Softsign()
            self.relu = torch.nn.ReLU()

        def forward(self,x):
            x = self.relu(self.embedding1(x))
            embed = self.embedding2(x)
            _,state = self.encoder(embed)
            state1 = (state[0] + state[1])/2.0
            context = state1.repeat(1,1,1)
            dec,_= self.decoder(context,state)
            dec = self.drop(dec)
            dec = self.mid_con(dec)
            dec = self.softsign(dec)
            output = self.to_v(dec)
            return output

    class Discriminate(nn.Module):
        def __init__(self):
            super(Discriminate, self).__init__()
            self.gru1 = nn.GRU(1, 32, num_layers=2)
            
            self.fc1 =nn.Linear(32, 1)

        def forward(self, x):
            x = x.unsqueeze(2)
            x = x.permute(1,0,2)
            _, x = self.gru1(x)
            x = x[-1,:,:]
            x = self.fc1(x)
            
            x = nn.functional.sigmoid(x)
            return x

    class StockDataSet(Dataset):
        def __init__(self, x, y=None, is_train=True):
            self.x = x
            self.is_train = is_train
            if self.is_train is True:
                self.y = y

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, item):
            return self.x[item], self.y[item] if self.is_train is True else self.x[item]

    def process_data(data, length=7):
        days = data.shape[0]
        x, y = [], []
        for i in range(days-length):
            x.append(data[i:i+length])
            y.append(data[i+length, pos_target])
        return x, y

    x_norm_scale = StandardScaler()
    y_norm_scale = StandardScaler()
    data_x, data_y = process_data(data)

    x = np.array(data_x).reshape(-1, 7, n_F).astype('float32')
    y = np.array(data_y).reshape(-1, 1).astype('float32')

    x = x_norm_scale.fit_transform(x.reshape(-1, n_F)).reshape(-1, 7, n_F)
    y = y_norm_scale.fit_transform(y)

    train_set = StockDataSet(x, y)
    train_data = DataLoader(train_set, batch_size=32, shuffle=True)

    g = Seq2Seq()
    d = Discriminate()
    his_loss = []
    lr = opt["lr"]
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr)
                    
    loss_func = nn.BCELoss()
    mse = nn.MSELoss()
    epoch = opt["epoch"]
    for epoch_id in range(epoch):
        for _, (x, y) in enumerate(train_data):
            
            real_labels = torch.ones(x.shape[0])
            fake_labels = torch.zeros(x.shape[0])
            sales_x = x[:, :, pos_target]
            real_xy = torch.cat([sales_x, y], dim=1)
            
            # 训练判别器
            d.train()
            g.eval()
            x = x.permute(1,0,2)
            fake_y = g(x.float()).reshape(-1, 1)
            fake_xy = torch.cat([sales_x, fake_y], dim=1)
            
            fake_out = d(fake_xy)
            d_fake_loss = loss_func(fake_out, fake_labels)
            
            real_out = d(real_xy)
            d_real_loss = loss_func(real_out, real_labels)
            
            d_loss = d_fake_loss + d_real_loss
            d.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g.train()
            d.eval()
            fake_y = g(x.float()).reshape(-1, 1)
            fake_xy = torch.cat([sales_x, fake_y], dim=1)
            fake_out = d(fake_xy)
            g_fake_loss = loss_func(fake_out, real_labels)
            g.zero_grad()
            g_fake_loss.backward()
            g_optimizer.step()
            
            loss = torch.sqrt(mse(fake_y.flatten(), y.flatten().float()))
            his_loss.append(loss.detach().numpy())

        if verbose == True:
            print("epoch ", epoch_id," : ",loss.detach().numpy(), d_loss.detach().numpy(), g_fake_loss.detach().numpy())

    predict_y = []
    for tmp_x in data_x:
        tmp = g(torch.autograd.Variable(torch.Tensor(x_norm_scale.transform(tmp_x).reshape(7, 1, n_F))))
        tmp = tmp.flatten().detach().numpy()
        tmp = y_norm_scale.inverse_transform(tmp.reshape(1,1))
        predict_y.append(tmp[0][0])
    if plot == True:
        fig = plt.figure(figsize=(12,4))
        plt.plot(range(len(data_y)), predict_y, label="predicted price")
        plt.plot(range(len(data_y)), data_y, label="real price")
        plt.title("prediction of stock price of "+stock)
        plt.xlabel("time")
        plt.ylabel("Closed Price")
        plt.legend(loc='best')
        plt.savefig("./data/result/"+stock+".png", format='png', dpi=400)
    return np.sqrt(mean_squared_error(data_y, predict_y)) / np.mean(data_y)