import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tcn import TCN, tcn_full_summary
from math import exp

class StockPricePrediction():
    def __init__(self,stock,period='max',model=None):
        self.stock=stock
        self.price=yf.Ticker(stock).history(period=period)
        self.price['log return'] = np.log(self.price.Close) - np.log(self.price.Close.shift(1))
        self.model=model
        self.model_period=0


    def fit(self,period=64, batch_size=5, timesteps=20, input_dim =1, epochs=20):
        try:
            if period>len(self.price):
                raise ValueError('Not enough stock price data. Please enter a shorter period.')
            
            return_list=[]
            for i in range(1,len(self.price)-period):
                return_list.append([])
                for day in range(period):
                    return_list[-1].append(self.price['log return'].iloc[i+day])
            return_list=pd.DataFrame(return_list)
            return_list=shuffle(return_list)
            X=return_list.iloc[:,:period-1].values
            y=return_list.iloc[:,period-1].values
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
            
            
            i = Input(batch_shape=(batch_size, timesteps, input_dim))

            o = TCN(return_sequences=False, dilations=[1, 2, 4, 8, 16, 32])(i)
            o = Dense(1)(o)
            
            m = Model(inputs=[i], outputs=[o])
            m.compile(optimizer='adam', loss='mse')
            
            tcn_full_summary(m, expand_residual_blocks=False)
            
            history = m.fit(X_train, y_train, epochs=20, validation_split=0.2)
            
            self.varience=m.evaluate(X_test, y_test, verbose=True)
            self.model=m
            self.model_period=period
            print('Model trained')
        except ValueError as e:
            print('ERROR:',repr(e))
        
        
    def predict(self,days='1'):
        try:
            if not self.model:
                raise ValueError('NO MODEL')
                
            return_list=[]
            for day in range(1,self.model_period+1):
                return_list.append(self.price['log return'][-day])
            return_list=pd.DataFrame(return_list[::-1]).values
            y_pred=self.model.predict(return_list)
            tmr_price=self.price['Close'][-1]*pow(exp(1),y_pred)
            return tmr_price
                
            
        except ValueError as e:
            print('ERROR:',repr(e))
            

if __name__=='__main__':
    s=StockPricePrediction('AAPL')
    s.fit()
    s.predict()
