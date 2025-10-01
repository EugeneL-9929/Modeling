import requests
import pandas as pd
import os

'''
Data Type:
    Core Stock API
        Granule: Daily (TIME_SERIES_DAILY)
        Size: 20 yrs
    Historial Options
        Function: HISTORICAL_OPTIONS
    Fx Rate
        Granule: Daily
        Function: CURRENCY_EXCHANGE_RATE
        from_currency
        to_currency
    US GDP
        Granule: Yearly
        Function: REAL_GDP
    Treasury Yield
        Granule: Monthly
        Function: TREASURY_YIELD
    CPI
        Granule: Monthly
        Function: CPI
    Company Overview
        Granule: Quarter
'''

class AlphaVantageStockData():
    def __init__(self, function='TIME_SERIES_DAILY', symbol='', outputsize='full', interval=''):
        self.baseUrl = 'https://www.alphavantage.co/query?'
        self.params = {
            'function' : function,
            '&symbol' : symbol,
            '&interval' : interval,
            '&outputsize' : outputsize,
            '&apikey' : 'K7G3VVX3F6EAFJL5',
        }
        self.url = '' + self.baseUrl
        for key, value in self.params.items():
            if value:
                self.url += key+'='+value
        self.rawData = {}
        self.symbol = symbol
        self.interval = interval
        print(f'visited following url: {self.url}')
        print(f'run getJson to initiate rawData')
        
    def timestampConvertor(self, time, timeZone='America/New_York'):
        timestamp = pd.Timestamp(time, tz=timeZone)
        unixTimestamp = int(timestamp.timestamp())
        return unixTimestamp

    def getJson(self):
        response = requests.get(self.url).json()
        for i in (response.keys()):
            if i.startswith('Time'):
                rawData = response[i]
        for key, value in rawData.items():
            time = self.timestampConvertor(key)
            data = {}
            titles = ['open', 'close', 'high', 'low', 'volume']
            for title, price in value.items():
                for i in titles:
                    if i in title:
                        data[i] = price
            self.rawData[str(time)] = data
    
    def getCsv(self):
        if self.rawData:
            time = list(self.rawData.keys())
            open = [value['open'] for _, value in self.rawData.items()]
            close = [value['close'] for _, value in self.rawData.items()]
            high = [value['high'] for _, value in self.rawData.items()]
            low = [value['low'] for _, value in self.rawData.items()]
            volume = [value['volume'] for _, value in self.rawData.items()]
            df = {'time':[], 'open' : [], 'close':[], 'high':[], 'low':[], 'volume':[]}
            if self.interval:
                csvName = f'Data/{self.symbol}_{self.interval}_stock_data.csv'
            else:
                csvName = f'Data/{self.symbol}_stock_data.csv'
            try:
                history = pd.read_csv(csvName, usecols=['time'], index_col='time')
                for i, j, k, l, m, n in zip(time, open, close, high, low, volume):
                    if i not in history.index:
                        df['time'].append(i)
                        df['open'].append(j)
                        df['close'].append(k)
                        df['high'].append(l)
                        df['low'].append(m)
                        df['volume'].append(m)
                    else:
                        print('repeat data')
                df = pd.DataFrame(df)
                df.to_csv(csvName, mode='a', index=False, header=not os.path.exists(f'Data/{self.symbol}_stock_data.csv'))
            except:
                df = pd.DataFrame({
                    'time' : time, 
                    'open' : open, 
                    'close' : close,
                    'high' : high,
                    'low' : low,
                    'volume' : volume,
                    })
                df.to_csv(csvName, index=False)
        else:
            print('please run getJson')
        

if __name__ == '__main__':
    avsd = AlphaVantageStockData(function='TIME_SERIES_DAILY', symbol='AAPL')
    avsd.getJson()
    avsd.getCsv()
