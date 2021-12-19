# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:30:22 2019

@author: tcx
"""
###shoujishuju
import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
nyyh=web.DataReader('GS','yahoo',dt.datetime(2010,1,1),dt.datetime(2018,12,31))
nyyh.tail()
print(type(nyyh))

data=pd.read_csv('dataGS.csv')
print(data)

###zuochushujufengexian
import warnings
import mxnet as mx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)
def parser(x):
 return dt.datetime.strptime(x,'%Y-%m-%d')
dataset_ex_df = pd.read_csv('dataGS.csv', header=0, parse_dates=[0], date_parser=parser)
dataset_ex_df[['Date', 'Close']].head(3)
print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))
plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset_ex_df['Date'], dataset_ex_df['Close'], label='Goldman Sachs stock')
plt.vlines(dt.date(2016,4, 20), 0, 270, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('USD')
plt.title('Figure 2: Goldman Sachs stock price')
plt.legend()
plt.show()



num_training_days = int(dataset_ex_df.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, 
 dataset_ex_df.shape[0]-num_training_days))


import math
def get_technical_indicators(dataset):
 # Create 7 and 21 days Moving Average
 dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
 dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
 
# # Create MACD
 dataset['26ema'] = pd.DataFrame.ewm(dataset['Close'], span=26).mean()
 dataset['12ema'] = pd.DataFrame.ewm(dataset['Close'], span=12).mean()
 dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
 # Create Bollinger Bands
 dataset['20sd'] = data['Close'].rolling(20).std()
 dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
 dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

 # Create Exponential moving average
 dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
 
 # Create Momentum
 dataset['momentum'] = dataset['Close']-1
 
 #Create log_momentum
 dataset['log_momentum'] = dataset['momentum'].apply(lambda x:math.log(x))
 
 return dataset
dataset_TI_df = get_technical_indicators(dataset_ex_df[['Close']]) #####jishuzhibiao
dataset_TI_df.head()


print(dataset_TI_df)###


def plot_technical_indicators(dataset, last_days):
 plt.figure(figsize=(16, 10), dpi=100)
 shape_0 = dataset.shape[0]
 xmacd_ = shape_0-last_days
 
 dataset = dataset.iloc[-last_days:, :]
 x_ = range(3, dataset.shape[0])
 x_ =list(dataset.index)
 
 # Plot first subplot
 plt.subplot(2, 1, 1)
 plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
 plt.plot(dataset['Close'],label='Closing Price', color='b')
 plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
 plt.plot(dataset['upper_band'],label='Upper Band', color='c')
 plt.plot(dataset['lower_band'],label='Lower Band', color='c')
 plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
 plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
 plt.ylabel('USD')
 plt.legend()
 # Plot second subplot
 plt.subplot(2, 1, 2)
 plt.title('MACD')
 plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
 plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
 plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
 plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')
 plt.legend()
 plt.show()
plot_technical_indicators(dataset_TI_df, 400)



data_FT = dataset_ex_df[['Date', 'Close']]
close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100]:
 fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
 plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(data_FT['Close'], label='Real')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
plt.legend()
plt.show()

from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime
series = data_FT['Close']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())  ####


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
plt.figure(figsize=(10, 7), dpi=80)
plt.show()


#
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
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
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#Plot the predicted (from ARIMA) and real prices
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('USD')
plt.title('Figure 5: ARIMA model on GS stock')
plt.legend()
plt.show()
#


#dataset_total_df.shape
#print('Total dataset has {} samples, and {} features.'.format(dataset_total_df.shape[0],dataset_total_df.shape[1]))
print(dataset_TI_df)

plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()

v=['ma7','ma21','26ema','12ema','MACD','20sd','upper_band','lower_band','ema','momentum','log_momentum']
import xgboost as xgb
def get_feature_importance_data(data_income,s):
 data = data_income.copy()
 y = data[s]
 X = data.iloc[:, 1:]
 
 train_samples = int(X.shape[0] * 0.65)
 
 X_train = X.iloc[:train_samples]
 X_test = X.iloc[train_samples:]
 y_train = y.iloc[:train_samples]
 y_test = y.iloc[train_samples:]
 
 return (X_train, y_train),(X_test, y_test)
# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'ma7')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
bar_width=0.2
fig = plt.figure(figsize=(8,8))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'ma21')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel2 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel2.feature_importances_))], xgbModel2.feature_importances_.tolist(), tick_label=X_test_FI.columns)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'20sd')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel3 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel3.feature_importances_))], xgbModel3.feature_importances_.tolist(), tick_label=X_test_FI.columns)



(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'Close')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel4 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel4.feature_importances_))], xgbModel4.feature_importances_.tolist(), tick_label=X_test_FI.columns)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'MACD')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel4 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel4.feature_importances_))], xgbModel4.feature_importances_.tolist(), tick_label=X_test_FI.columns)

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'26ema')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel5 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel5.feature_importances_))], xgbModel5.feature_importances_.tolist(), tick_label=X_test_FI.columns)



(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'12ema')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel6 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel6.feature_importances_))], xgbModel6.feature_importances_.tolist(), tick_label=X_test_FI.columns)

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'upper_band')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel7 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel7.feature_importances_))], xgbModel7.feature_importances_.tolist(), tick_label=X_test_FI.columns)

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'lower_band')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel8 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel8.feature_importances_))], xgbModel8.feature_importances_.tolist(), tick_label=X_test_FI.columns)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'ema')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel9 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel9.feature_importances_))], xgbModel9.feature_importances_.tolist(), tick_label=X_test_FI.columns)

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'momentum')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel10 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel10.feature_importances_))], xgbModel10.feature_importances_.tolist(), tick_label=X_test_FI.columns)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df,'log_momentum')
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel11 = regressor.fit(X_train_FI,y_train_FI, 
 eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)],
 verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))
plt.bar([i for i in range(len(xgbModel11.feature_importances_))], xgbModel11.feature_importances_.tolist(), tick_label=X_test_FI.columns)


plt.title('Figure 6: Feature importance of the technical indicators.')
plt.show()
print(dataset_TI_df['20sd'])
print(X_test_FI.columns)






def gelu(x):
 return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
def relu(x):
 return max(x, 0)
def lrelu(x):
 return max(0.01*x, x)


plt.figure(figsize=(15, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)
ranges_ = (-10, 3, .25)
plt.subplot(1, 2, 1)
plt.plot([i for i in np.arange(*ranges_)], [relu(i) for i in np.arange(*ranges_)], label='ReLU', marker='.')
plt.plot([i for i in np.arange(*ranges_)], [gelu(i) for i in np.arange(*ranges_)], label='GELU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.title('Figure 7: GELU as an activation function for autoencoders')
plt.ylabel('f(x) for GELU and ReLU')
plt.xlabel('x')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot([i for i in np.arange(*ranges_)], [lrelu(i) for i in np.arange(*ranges_)], label='Leaky ReLU')
plt.hlines(0, -10, 3, colors='gray', linestyles='--', label='0')
plt.ylabel('f(x) for Leaky ReLU')
plt.xlabel('x')
plt.title('Figure 8: LeakyReLU')
plt.legend()
plt.show()



#from utils import *
#import time
#import numpy as np
#from mxnet import nd, autograd, gluon
#from mxnet.gluon import nn, rnn
#import mxnet as mx
#import datetime
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#import math
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
#from sklearn.metrics import accuracy_score
#import warnings
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import time
VAE_data=dataset_TI_df
batch_size = 64
n_batches = VAE_data.shape[0]/batch_size
VAE_data = VAE_data.values
train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days,:-1]}, 
 label={'label': VAE_data[:num_training_days, -1]}, batch_size = batch_size)
test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:,:-1]}, 
 label={'label': VAE_data[num_training_days:,-1]}, batch_size = batch_size)
model_ctx = mx.cpu()
class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, 
    batch_size=100, act_type='gelu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        super(VAE, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')
 
        for i in range(n_layers):
            self.encoder.add(nn.Dense(n_hidden, activation=act_type))
            self.encoder.add(nn.Dense(n_latent*2, activation=None))
            self.decoder = nn.HybridSequential(prefix='decoder')
        for i in range(n_layers):
            self.decoder.add(nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(nn.Dense(n_output, activation='sigmoid'))
    def hybrid_forward(self, F, x):
        h = self.encoder(x)
        print(h)
        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
        z = mu + F.exp(0.5*lv)*eps
        y = self.decoder(z)
        self.output = y
        KL = 0.5*F.sum(1+lv-mu*mu-F.exp(lv),axis=1)
        logloss = F.sum(x*F.log(y+self.soft_zero)+ (1-x)*F.log(1-y+self.soft_zero), axis=1)
        loss = -logloss-KL
        return loss
n_hidden=400 # neurons in each layer
n_latent=2 
n_layers=3 # num of dense layers in encoder and decoder respectively
n_output=VAE_data.shape[1]-1 

net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size, act_type='relu')
net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})
print(net)
#
#
#batch.data=dataset_TI_df

n_epoch = 150
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []
for epoch in range(n_epoch):
     epoch_loss = 0
     epoch_val_loss = 0
    
     train_iter.reset()
     test_iter.reset()
    
     n_batch_train = 0
     for batch in train_iter:
         n_batch_train +=1
         data = batch.data[0].as_in_context(mx.cpu())
         
         
         with autograd.record():
             loss = net(data)
         loss.backward()
         trainer.step(data.shape[0])
         epoch_loss += nd.mean(loss).asscalar()
     
     n_batch_val = 0
     for batch in test_iter:
         n_batch_val +=1
         data = batch.data[0].as_in_context(mx.cpu())
         loss = net(data)
         epoch_val_loss += nd.mean(loss).asscalar()
     
     epoch_loss /= n_batch_train
     epoch_val_loss /= n_batch_val
     
     training_loss.append(epoch_loss)
     validation_loss.append(epoch_val_loss)
     
     """if epoch % max(print_period, 1) == 0:
         print('Epoch {}, Training loss {:.2f}, Validation loss {:.2f}'.
             format(epoch, epoch_loss, epoch_val_loss))"""

end = time.time()
print('Training completed in {} seconds.'.format(int(end-start)))
#
dataset_TI_df['Date'] = dataset_ex_df['Date']
vae_added_df = mx.nd.array(dataset_TI_df.iloc[:, :-1].values)
print('The shape of the newly created (from the autoencoder) features is {}.'.format(vae_added_df.shape))
print(vae_added_df.shape)
#vae_added_df=np.asarray(vae_added_df,'float32')
#print(vae_added_df)
#
#vae_added_df.fillna(method='ffill',axis=1)
#print(vae_added_df)
#vae_added_df.fillna(method='ffill',axis=0)

#u=[]
#for i in range(len(vae_added_df)):
#    p=[]
#    for j in range(len(vae_added_df[i])):
#        p.append(vae_added_df[i][j])
#    u.append(p)
#print(u[0][0])
#
#vae_added_df=u
#
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#pca = PCA(n_components=.8)
##vae_added_df = np.concatenate(vae_added_df, axis=0) 
#x_pca = StandardScaler().fit_transform(vae_added_df)
#principalComponents = pca.fit_transform(x_pca)
#principalComponents.n_components_
#print(x_pca)
#print(principalComponents)

gan_num_features = vae_added_df.shape[1]
sequence_length = 17
class RNNModel(gluon.Block):
     def __init__(self, num_embed, num_hidden, num_layers, bidirectional=False, 
                  sequence_length=sequence_length, **kwargs):
         super(RNNModel, self).__init__(**kwargs)
         self.num_hidden = num_hidden
         with self.name_scope():
             self.rnn = rnn.LSTM(num_hidden, num_layers, input_size=num_embed, 
                                 bidirectional=bidirectional, layout='TNC')
 
             self.decoder = nn.Dense(1, in_units=num_hidden)
 
     def forward(self, inputs, hidden):
         output, hidden = self.rnn(inputs, hidden)
         decoded = self.decoder(output.reshape((-1, self.num_hidden)))
         return decoded, hidden
 
     def begin_state(self, *args, **kwargs):
         return self.rnn.begin_state(*args, **kwargs)
 
lstm_model = RNNModel(num_embed=gan_num_features, num_hidden=500, num_layers=1)
lstm_model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
trainer = gluon.Trainer(lstm_model.collect_params(), 'adam', {'learning_rate': .01})
loss = gluon.loss.L1Loss()

print(lstm_model)


class TriangularSchedule():
     def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5): 
         self.min_lr = min_lr
         self.max_lr = max_lr
         self.cycle_length = cycle_length
         self.inc_fraction = inc_fraction
 
     def __call__(self, iteration):
         if iteration <= self.cycle_length*self.inc_fraction:
             unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
         elif iteration <= self.cycle_length:
             unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
         else:
             unit_cycle = 0
         adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
         return adjusted_cycle
class CyclicalSchedule():
     def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
         self.schedule_class = schedule_class
         self.length = cycle_length
         self.length_decay = cycle_length_decay
         self.magnitude_decay = cycle_magnitude_decay
         self.kwargs = kwargs
 
     def __call__(self, iteration):
         cycle_idx = 0
         cycle_length = self.length
         idx = self.length
         while idx <= iteration:
             cycle_length = math.ceil(cycle_length * self.length_decay)
             cycle_idx += 1
             idx += cycle_length
         cycle_offset = iteration - idx + cycle_length
 
         schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
         return schedule(cycle_offset) * self.magnitude_decay**cycle_idx
schedule = CyclicalSchedule(TriangularSchedule, min_lr=0.5, max_lr=2, cycle_length=500)
iterations=1500
plt.plot([i+1 for i in range(iterations)],[schedule(i) for i in range(iterations)])
plt.title('Learning rate for each epoch')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()
    
num_fc = 512
# ... other parts of the GAN
cnn_net = gluon.nn.Sequential()
with net.name_scope():
 
 # Add the 1D Convolutional layers
     cnn_net.add(gluon.nn.Conv1D(32, kernel_size=5, strides=2))
     cnn_net.add(nn.LeakyReLU(0.01))
     cnn_net.add(gluon.nn.Conv1D(64, kernel_size=5, strides=2))
     cnn_net.add(nn.LeakyReLU(0.01))
     cnn_net.add(nn.BatchNorm())
     cnn_net.add(gluon.nn.Conv1D(128, kernel_size=5, strides=2))
     cnn_net.add(nn.LeakyReLU(0.01))
     cnn_net.add(nn.BatchNorm())
 
 # Add the two Fully Connected layers
     cnn_net.add(nn.Dense(220, use_bias=False), nn.BatchNorm(), nn.LeakyReLU(0.01))
     cnn_net.add(nn.Dense(220, use_bias=False), nn.Activation(activation='relu'))
     cnn_net.add(nn.Dense(1))
 
# ... other parts of the GAN
print(cnn_net)
#class GAN():
#    def _init_(self):
#        optimizer=Adam(0.0002,0.5)
#        self.discrimniator=self.build_discriminator()
#        self.discriminator.compile(loss='binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
#        
#        self.generator=self.build_generator()
#        z=vae_added_df
#        img=self.generator(z)
#        self.discriminiator.trainable=False
#        
#        validity=self.discrimniator(lstm_model)
#        self.combined=Model(z,validity)
#        self.combined.compile(loss='binary_crossentropy',optimizer=optimizer)
#    def build_generator(self):
#        model=lstm_model
#        noise=vae_added_df
#        img=model(noise)
#        return Model(noise,img)
#    def build_discriminator(self):
#        model=cnn_net
#        img=vae_added_df
#        validity=model(img)
#        return Model(img,validity)
#    def train(self, epochs, batch_size=128, sample_interval=50):
#
#        (X_train, _), (_, _) = mnist.vae_added_df()
#        X_train = X_train / 127.5 - 1.
#        X_train = np.expand_dims(X_train, axis=3)
#        valid = np.ones((batch_size, 1))
#        fake = np.zeros((batch_size, 1))
#        for epoch in range(epochs):
#            idx = np.random.randint(0, X_train.shape[0], batch_size)
#            imgs = X_train[idx]
#            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#            gen_imgs = self.generator.predict(noise)
#            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
#            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#            g_loss = self.combined.train_on_batch(noise, valid)
#            if epoch % sample_interval == 0:
#                self.sample_images(epoch)
#    def images(self, epoch):
#        r, c = 5, 5
#        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
#        gen_imgs = self.generator.predict(noise)
#        gen_imgs = 0.5 * gen_imgs + 0.5
#        fig, axs = plt.subplots(r, c)
#        cnt = 0
#        for i in range(r):
#            for j in range(c):
#                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap=cmap)
#                axs[i,j].axis('off')
#                cnt += 1
#        fig.savefig("images/%d.png" % epoch)
#        plt.figure(figsize=(16, 12))
#        plt.plot(y_test, label='test')
#        plt.plot(y_pred, label='pred')
#        plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
#        plt.show()
#        plt.close()
#        
#
#if __name__ == '__main__':
#    gan = GAN()
#    gan.train(epochs=30000, batch_size=32, sample_interval=200)
#print(gan)        

#用sklearn和GAN似乎有点算不动。。。尝试新的用tensorflow的架构
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.framework import ops
data=pd.read_csv('dataGS.csv')
data.drop('Date',axis=1,inplace=True)
print(data)
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]
input_dim = X_train.shape[1]
output_dim = 1
hidden_1 = 1024
hidden_2 = 512
hidden_3 = 256
hidden_4 = 128
batch_size = 256    
epochs = 10     
ops.reset_default_graph()
X = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
Y = tf.placeholder(shape=[None], dtype=tf.float32)
W1 = tf.get_variable('W1', [input_dim, hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable('b1', [hidden_1], initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2', [hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable('b2', [hidden_2], initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3', [hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable('b3', [hidden_3], initializer=tf.zeros_initializer())
W4 = tf.get_variable('W4', [hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable('b4', [hidden_4], initializer=tf.zeros_initializer())
W5 = tf.get_variable('W5', [hidden_4, output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b5 = tf.get_variable('b5', [output_dim], initializer=tf.zeros_initializer())
h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
h4 = tf.nn.relu(tf.add(tf.matmul(h3, W4), b4))
out = tf.transpose(tf.add(tf.matmul(h4, W5), b5))
loss = tf.reduce_mean(tf.squared_difference(out, Y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        for i in range(y_train.shape[0] // batch_size):
            start = i * batch_size
            batch_x = X_train[start : start + batch_size]
            batch_y = y_train[start : start + batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            if i % 50 == 0:
                print('MSE Train:', sess.run(loss, feed_dict={X: X_train, Y: y_train}))
                print('MSE Test:', sess.run(loss, feed_dict={X: X_test, Y: y_test}))
                y_pred = sess.run(out, feed_dict={X: X_test})
                y_pred = np.squeeze(y_pred)
                plt.figure(figsize=(16, 12))
                plt.plot(y_test, label='test')
                plt.plot(y_pred, label='pred')
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                plt.legend()
                plt.show()
                