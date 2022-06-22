'''
This is a chart creation tool for Ehler Plots.
Ehler plots are essentially a rotation plot that compares two assets against each other.

This program allows to create such Ehler plots with assets from yahoo finance or even self calculated spreads.

Created by Korbinian Gabriel for the Macro Summary
'''

# Libraries
import pandas as pd
from scipy import stats
import numpy as np
import math

import yfinance as yf

import datetime as dt

import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import seaborn as sns


# Main Class
class EhlerPlot(object):
    """This class creates the ehler plots. It can be called in any other bigger program."""
    
    def __init__(self, ticker1, ticker2, start_date='1950-01-01', freq='D', title='-'):
        """Initializes class Object

        Args:
            ticker1 (string): Ticker name from yahoo finance
            ticker2 (string): ticker name from yahoo finance
            start_date (str, optional): Start date to download data. Default enter '1950-01-01'.
            freq (str, optional): Frequency in D, W or M. Default enter 'D'.
            title (str, optional): Optional title for the graphic. Default enter '-'.
        """
        self.input = [ticker1, ticker2]
        self.tickers = []
        self.start_date = start_date
        self.title = title
        self.data = None
        self.ticker1_data = None
        self.ticker2_data = None
        self.ticker_data = None
        self.freq = freq
        
    def analyze_tickers(self):
        """Analyses whether the ticker is a spread.
        """
        for ticker in self.input:
            if "/" in ticker:
                self.tickers.append(ticker.split("/")[0])
                self.tickers.append(ticker.split("/")[1])
                print(f"Spread ticker split for {ticker}.")
            elif "-" in ticker:
                self.tickers.append(ticker.split("-")[0])
                self.tickers.append(ticker.split("-")[1])
                print(f"Spread ticker split for {ticker}.")
            else:
                self.tickers.append(ticker)
            
    def download_data(self):
        """Downloads price data from yahoo finance.
        """
        self.data = yf.download(self.tickers, start_date=self.start_date)['Close'].dropna()
    
    def spread(self):
        """Calculates spread if spread is given.
        """
        if len(self.input) < len(self.tickers):
            print("Calculating spreads...")
            for ticker in self.input:
                if "/" in ticker:
                    self.data[ticker] = self.data[ticker.split("/")[0]] / self.data[ticker.split("/")[1]]
                elif "-" in ticker:
                    self.data[ticker] = self.data[ticker.split("-")[0]] - self.data[ticker.split("-")[1]]
            self.data = self.data[self.input]
            print("Spreads calculated.")
            print(self.data.tail())
        
                     
    def resampling(self):
        """Resamples data if frequency is different."""
        if self.freq != "D":
            self.data = self.data.resample(self.freq).agg('first')
            print(f'Resampling data to {self.freq}.')
            
            
    def indicator(self, LPPeriod = 20, HPPeriod = 125):
        """Creates ehler plot indicator.

        Args:
            LPPeriod (int, optional): Low pass filter period. Defaults to 20.
            HPPeriod (int, optional): High pass filter period. Defaults to 125.
        """
        data = self.data.copy()
        data.reset_index(inplace=True)
        deg2rad = math.pi / 180.0
        hpa1 = math.exp(-1.414 * math.pi / HPPeriod)
        hpb1 = 2.0 * hpa1 * math.cos((1.414 * 180.0 / HPPeriod) * deg2rad)
        hpc2 = hpb1
        hpc3 = -hpa1 * hpa1
        hpc1 = (1 + hpc2 - hpc3) / 4
        ssa1 = math.exp(-1.414 * math.pi / LPPeriod)
        ssb1 = 2.0 * ssa1 * math.cos((1.414 * 180.0 / LPPeriod) * deg2rad)
        ssc2 = ssb1
        ssc3 = -ssa1 * ssa1
        ssc1 = 1 - ssc2 - ssc3
        
        data['lstvaluesp'] = np.nan
        data['lstvaluesp2'] = np.nan
        
        data['hp1'] = 0.0
        data['hp2'] = 0.0
        
        for index, row in data.iterrows():
            if index>2:
                data.loc[index, 'hp1'] = hpc1 * (data.loc[index, self.input[0]] - 2 * data.loc[index-1, self.input[0]] + data.loc[index-2, self.input[0]]) + hpc2 * data.loc[index-1, 'hp1'] + hpc3 * data.loc[index-2, 'hp1']
                data.loc[index, 'hp2'] = hpc1 * (data.loc[index, self.input[1]] - 2 * data.loc[index-1, self.input[1]] + data.loc[index-2, self.input[1]]) + hpc2 * data.loc[index-1, 'hp2'] + hpc3 * data.loc[index-2, 'hp2']
            
        data['price'] = 0.0
        data['price2'] = 0.0
        
        for index, row in data.iterrows():
            if index>2:
                data.loc[index, 'price'] = ssc1 * (data.loc[index, 'hp1'] + data.loc[index-1, 'hp1']) / 2 + ssc2 * data.loc[index-1, 'price'] + ssc3 * data.loc[index-2, 'price']
                data.loc[index, 'price2'] = ssc1 * (data.loc[index, 'hp2'] + data.loc[index-1, 'hp2']) / 2 + ssc2 * data.loc[index-1, 'price2'] + ssc3 * data.loc[index-2, 'price2']
            
        data['pricems'] = 0.0
        data['pricerms'] = 0.0
        data['price2ms'] = 0.0
        data['price2rms'] = 0.0
        
        for index, row in data.iterrows():
            if index < 2:
                data.loc[index, 'pricems'] = math.pow(data.loc[index, 'price'], 2)
                data.loc[index, 'price2ms'] = math.pow(data.loc[index, 'price2'], 2)
            else:
                data.loc[index, 'pricems'] = 0.0242 * data.loc[index, 'price'] * data.loc[index, 'price'] + 0.9758 * data.loc[index-1, 'pricems']
                data.loc[index, 'price2ms'] = 0.0242 * data.loc[index, 'price2'] * data.loc[index, 'price2'] + 0.9758 * data.loc[index-1, 'price2ms']
                
            if data.loc[index, 'pricems'] != 0:
                data.loc[index, 'pricerms'] = data.loc[index, 'price'] / math.sqrt(data.loc[index, 'pricems'])
            if data.loc[index, 'price2ms'] != 0:
                data.loc[index, 'price2rms'] = data.loc[index, 'price2'] / math.sqrt(data.loc[index, 'price2ms'])
            
            data.loc[index, 'lstvaluesp'] = data.loc[index, 'pricerms']
            data.loc[index, 'lstvaluesp2'] = data.loc[index, 'price2rms']
        
        data.set_index('Date', inplace=True)
        data = data.tail(300)
        data[self.input] = data[self.input] / data[self.input].iloc[0]
        self.data = data
        
        
    def plot(self):
        """Generates plots.
        """
        fig, ax = plt.subplots(4, 2, sharex=True, figsize=(15, 10))
        ax[0, 0].set_title(self.input[0])
        ax[0, 0].plot(self.data[self.input[0]])
        ax[1, 0].set_title('price')
        ax[1, 0].plot(self.data['price'])
        ax[2, 0].set_title('pricems')
        ax[2, 0].plot(self.data['pricems'])
        ax[3, 0].set_title('pricerms')
        ax[3, 0].plot(self.data['pricerms'])
        
        ax[0, 1].set_title(self.input[1])
        ax[0, 1].plot(self.data[self.input[1]])
        ax[1, 1].set_title('price2')
        ax[1, 1].plot(self.data['price2'])
        ax[2, 1].set_title('price2ms')
        ax[2, 1].plot(self.data['price2ms'])
        ax[3, 1].plot(self.data['price2rms'])
        ax[3, 1].set_title('price2rms')
        ax[0, 1].text(0.22, 0.05, 'Source: Macro Summary, Yahoo Finance.', fontsize=7, alpha=0.5, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        plt.draw()   
         
        fig, ax = plt.subplots(2)
        ax[0].plot(self.data[self.input[0]])
        ax[0].plot(self.data[self.input[1]])
        ax[0].set_title(self.title)
        color = ['b'] * (len(self.data.tail(40)) - 1) + ['r']
        ax[1].scatter(x=self.data['lstvaluesp'].tail(40), y=self.data['lstvaluesp2'].tail(40), color=color)
        ax[1].plot(self.data['lstvaluesp'].tail(40), self.data['lstvaluesp2'].tail(40))
        ax[1].set_ylabel(self.input[0])
        ax[1].set_xlabel(self.input[1])
        ax[1].text(0.22, 0.05, 'Source: Macro Summary, Yahoo Finance.', fontsize=7, alpha=0.5, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        plt.show()
        
        
def demo():
    """Runs a demo program.
    """
    EP = EhlerPlot(ticker1='SPY', ticker2='RTX')
    EP.analyze_tickers()
    EP.download_data()
    EP.spread()
    EP.resampling()
    EP.indicator()
    print(EP.data)
    EP.plot()
    print('done.')

def run():
    """Runs the full console program with questions.
    """
    print('')
    print('-'*60)
    print('-'*60)
    print('Ehler Chart Program')
    print('Date: ', dt.datetime.today().date())
    print('-'*60)
    
    print('Running correlation scatter analysis tool....')
    print('Please select "correlation plotter" to run the correlation plotting tool.')
    
    program = input('Select "ehler charts" to run the program or "demo" to run demo: ')
    
    if program == "ehler charts":
        print('Please enter these details from yahoo finance for the asset you want to add:')
        ticker1 = input('Ticker 1 (x-axis): '), 
        ticker2 = input('Ticker 2 (y-axis): '), 
        start_date = input('Start date of data (e.g. default 1950-01-01): ' ), 
        freq = input('Frequency of time series (default: D): '),
        title = input('Set a chart title: '), 
        print(f"Inputs: ticker1: {ticker1}, ticker2: {ticker2}, start_date: {start_date}, title: {title}, type: {type}.")
        
        EP = EhlerPlot(ticker1 = str(ticker1[0]), 
                     ticker2 = str(ticker2[0]), 
                     start_date = str(start_date[0]), 
                     freq = str(freq[0])
                     )

        print("Analyzing tickers...")
        EP.analyze_tickers()
        
        print('Downloading data...')
        EP.download_data()
        
        EP.spread()
        
        EP.resampling()
        
        print('Calculating...')
        EP.indicator()
        
        print('Plotting...')    
        EP.plot()
        
    elif program == "demo":
        demo()
    else:
        print('No function selected.')
        print('Please select a function: update db, add symbol.')
        print('Closing program.')    
    
    
if __name__ == '__main__':
    run()
    
        
        
            
                
            
        
        
        
        
