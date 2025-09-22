import requests
import json

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
'''

class AlphaVantageData():
    def __init__(self, function='TIME_SERIES_DAILY', interval='daily', symbol='', from_currency='', to_currency=''):
        self.baseUrl = 'https://www.alphavantage.co/query?'
        self.params = {
            'function' : function,
            '&symbol' : symbol,
            '&interval' : interval,
            '&from_currency' : from_currency,
            '&to_currency' : to_currency,
            '&apikey' : 'K7G3VVX3F6EAFJL5',
        }
        self.url = '' + self.baseUrl
        for key, value in self.params.items():
            if not value:
                self.url += key+'='+value
        self.rawData = []
        print(f'visited following url: {self.url}')
        print(f'run getJson to initiate rawData')
        
        def getJson(self):
            response = requests.get(self.url).json()
            if response['data']:
                self.rawData
