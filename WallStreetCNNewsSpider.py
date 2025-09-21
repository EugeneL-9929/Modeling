# pip install requets
# pip install selenium
# pip install beautifulsoup4 html5lib
# pip install openai

import re
import os
import requests
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import datetime
import pytz
import pandas as pd

headers = {
    'User-Agent': ''
}


class WallStreetNews():
    def __init__(self, channel='global', cursor='', limit=20):
        self.baseUrl = 'https://api-one-wscn.awtmt.com/apiv1/content/information-flow?'
        self.params = {
            'channel': channel,
            '&accept': 'article',
            '&cursor': cursor,
            '&limit': str(limit),
            '&action': 'upglide'
        }
        self.url = '' + self.baseUrl
        for key, value in self.params.items():
            self.url += key + '=' + value
        print(f"visited following url: {self.url}")
        self.nextCursor = ''
        self.rawData = []
        print(f"run getJson to initiate nextCursor and returnData")

    def getJson(self):
        response = requests.get(self.url).json()
        if response['code'] == 20000:
            self.nextCursor = response['data']['next_cursor']
            self.rawData = [i['resource'] for i in response['data']['items'] if i['resource_type'] == 'article']
        else:
            print(f"failed to load! status code: {response['code']}")

    def getNewJson(self):
        self.params['&cursor'] = self.nextCursor
        self.url = self.baseUrl
        for key, value in self.params.items():
            if key == '&cursor':
                value = value.replace('=', '%3D')
            self.url += key + '=' + value
        print(f"visited following url: {self.url}")
        response = requests.get(self.url).json()
        if response['code'] == 20000:
            if self.nextCursor != response['data']['next_cursor']:
                self.nextCursor = response['data']['next_cursor']
            else:
                print('reaching the maximum.')
            self.rawData = [i['resource'] for i in response['data']['items'] if i['resource_type'] == 'article']
        else:
            print(f"failed to load! status code: {response['code']}")

    def getAuthors(self):
        if self.rawData:
            authorsList = [i['author']['display_name'] for i in self.rawData]
            return authorsList
        else:
            print('please initiate!')

    def getSummary(self):
        if self.rawData:
            summaryList = [i['content_short'] for i in self.rawData]
            return summaryList
        else:
            print('please initiate!')

    def getTitles(self):
        if self.rawData:
            titlesList = [i['title'] for i in self.rawData]
            return titlesList
        else:
            print('please initiate!')

    def getIds(self):
        if self.rawData:
            idsList = [i['id'] for i in self.rawData]
            return idsList
        else:
            print('please initiate!')

    def getTimes(self):
        if self.rawData:
            timesList = [i['display_time'] for i in self.rawData]
            return timesList
        else:
            print('please initiate!')

    def nyTimeZone(self, timestamp):
        time = datetime.datetime.fromtimestamp(timestamp)
        ny_tz = pytz.timezone('America/New_York')
        time = time.astimezone(ny_tz)
        return time.strftime('%Y-%m-%d %H:%M')

    def cnTimeZone(self, timestamp):
        time = datetime.datetime.fromtimestamp(timestamp)
        cn_tz = pytz.timezone('Asia/Shanghai')
        time = time.astimezone(cn_tz)
        return time.strftime('%Y-%m-%d %H:%M')

    def jpTimeZone(self, timestamp):
        time = datetime.datetime.fromtimestamp(timestamp)
        jp_tz = pytz.timezone('Asia/Tokyo')
        time = time.astimezone(jp_tz)
        return time.strftime('%Y-%m-%d %H:%M')


class NewsDetails():
    def __init__(self, id):
        self.baseUrl = "https://api-one-wscn.awtmt.com/apiv1/content/articles/{}?extract=0"
        self.id = id
        self.details = ''
        print(f"run getDetails to initiate details")

    def getDetails(self):
        response = requests.get(self.baseUrl.format(id), headers=headers).json()
        details = response.encode().decode('utf-8')
        details = re.sub('<[^>]+>', ' ', details)
        details = re.sub(r'\s+', ' ', details)
        self.details = details


class Translator():
    def __init__(self, extraConstraint=''):
        self.baseUrl = 'https://api.poe.com/v1/chat/completions'
        self.apiKey = 'YYd3phri6tU32JTJwV6BaJGUUBasFpi8h__VTt22VJA'
        self.headers = {
            'Authorization': f'Bearer {self.apiKey}',
            'Content-Type': 'application/json'
        }
        self.messages = [{'role': 'system', 'content': 'you are a helpful assistant specialized in finance.'},
                         {'role': 'user',
                          'content': 'when i send you one chinese text, help me translate into the english, ' + extraConstraint}]
        self.initializationData = {
            'model': 'gpt-3.5-turbo',
            'messages': self.messages
        }
        try:
            response = requests.post(self.baseUrl, headers=self.headers, json=self.initializationData)
            print(f'initialization status code: {response.status_code}')
            results = response.json()['choices'][0]['message']['content']
            print(f'initialization results: {results}')
            self.messages.append({'role': 'assistant', 'content': results})
        except requests.exceptions.RequestException as e:
            print(f'request failed: {e}')
        except Exception as e:
            print(f'error msg: {e}')

    def translate(self, content):
        user_messages = self.messages + [{'role': 'user', 'content': content}]
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': user_messages
        }
        response = requests.post(self.baseUrl, headers=self.headers, json=data)
        print(f'status code: {response.status_code}')
        translated = response.json()['choices'][0]['message']['content']
        return translated


def operationSpiderAbstractsEN(limit=50, channel='global', loadTimes=1):
    wsn = WallStreetNews(limit=limit, channel=channel)
    wsn.getJson()
    trans = Translator('no extra text return except the content i send')
    for i in range(loadTimes):
        if i != 0:
            wsn.getNewJson()
        df = {'time': [], 'title': [], 'abstract': [], 'id': [], 'author': []}
        timesList = wsn.getTimes()
        timesList = [wsn.nyTimeZone(i) for i in timesList]
        summaryList = wsn.getSummary()
        idsList = wsn.getIds()
        titlesList = wsn.getTitles()
        authorsList = wsn.getAuthors()
        try:
            history = pd.read_csv('news_data.csv', usecols=['time'], index_col='time')
            for i, j, k, l, m in zip(timesList, summaryList, idsList, titlesList, authorsList):
                if i not in history.index:
                    df['time'].append(i)
                    df['abstract'].append(trans.translate(j))
                    df['id'].append(k)
                    df['title'].append(trans.translate(l))
                    df['author'].append(m)
                else:
                    print('repeat data')
            df = pd.DataFrame(df)
            df.to_csv('news_data.csv', mode='a', index=False, header=not os.path.exists('news_data.csv'))
        except FileNotFoundError as e:
            df = pd.DataFrame(
                {'time': timesList, 'title': titlesList, 'abstract': summaryList, 'id': idsList, 'author': authorsList})
            df.to_csv('news_data.csv', index=False, encoding='utf-8-sig')


def operationSpiderAbstractsCN(limit=50, channel='global', loadTimes=1):
    wsn = WallStreetNews(limit=limit, channel=channel)
    wsn.getJson()
    for i in range(loadTimes):
        if i != 0:
            wsn.getNewJson()
        df = {'time': [], 'title': [], 'abstract': [], 'id': [], 'author': []}
        timesList = wsn.getTimes()
        timesList = [wsn.nyTimeZone(i) for i in timesList]
        summaryList = wsn.getSummary()
        idsList = wsn.getIds()
        titlesList = wsn.getTitles()
        authorsList = wsn.getAuthors()
        try:
            history = pd.read_csv('news_data_cn.csv', usecols=['time'], index_col='time')
            for i, j, k, l, m in zip(timesList, summaryList, idsList, titlesList, authorsList):
                if i not in history.index:
                    df['time'].append(i)
                    df['abstract'].append(j)
                    df['id'].append(k)
                    df['title'].append(l)
                    df['author'].append(m)
                else:
                    print('repeat data')
            df = pd.DataFrame(df)
            df.to_csv('news_data_cn.csv', mode='a', index=False, header=not os.path.exists('news_data_cn.csv'))
        except FileNotFoundError as e:
            df = pd.DataFrame(
                {'time': timesList, 'title': titlesList, 'abstract': summaryList, 'id': idsList, 'author': authorsList})
            df.to_csv('news_data_cn.csv', index=False, encoding='utf-8-sig')


######################################################################################
def getHtmlListScrollLoading(scrollTimes=5, scrollPauseTime=5):
    driver = webdriver.Chrome()
    url = 'https://wallstreetcn.com/news/global'
    driver.get(url)
    initialHeight = driver.execute_script('return document.body.scrollHeight')
    print("initial page length: " + str(initialHeight))
    for i in range(scrollTimes):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(scrollPauseTime)  # wait for loading page
        newHeight = driver.execute_script('return document.body.scrollHeight')
        print('updated page length: ' + str(newHeight))
    htmlListContent = driver.page_source
    return htmlListContent


def parseHtmlList(htmlListContent):
    soup = BeautifulSoup(htmlListContent, 'html.parser')
    container = soup.select('div.article-entry div.container')
    anchor = [i.find('a') for i in container]
    meta = [i.find('div', class_='meta') for i in container]
    newsLinks = [i['href'] for i in anchor]
    titles = [i.find('span').text for i in anchor]
    abstracts = [i.find('div', class_='content').text for i in container]
    authors = [i.find('div', class_='author').text for i in meta]
    times = [i.find('time')['datetime'] for i in meta]
    ids = [i.split('/')[-1] for i in newsLinks]
    returnValue = {
        "links": newsLinks,
        "titles": titles,
        "authors": authors,
        "times": times,
        "abstracts": abstracts,
        "ids": ids
    }
    return returnValue


def getHtml(newsId):
    url = "https://api-one-wscn.awtmt.com/apiv1/content/articles/{}?extract=0"
    response = requests.get(url.format(newsId), headers=headers).json()
    abstract = response['data']['content_short']
    # print(abstract)
    return abstract


def translator(contentsList):
    translatedContentList = []
    apiKey = 'YYd3phri6tU32JTJwV6BaJGUUBasFpi8h__VTt22VJA'
    url = "https://api.poe.com/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {apiKey}',
        'Content-Type': 'application/json'
    }
    for i in contentsList:
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'user', 'content': i + 'translate this stock market news into english'}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            translatedContentList.append(response.json()['choices'][0]['message']['content'])
        else:
            translatedContentList.append("failed to translate this piece of message")
            print(f"request failed: {response.status_code}, error msg: {response.text}")
    return translatedContentList


if __name__ == '__main__':
    # operationSpiderAbstractsCN(loadTimes=50)
    operationSpiderAbstractsEN(loadTimes=3)