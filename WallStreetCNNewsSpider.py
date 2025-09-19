# pip install requets
# pip install selenium
# pip install beautifulsoup4 html5lib
# pip install googletrans
import re
import requests
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from googletrans import Translator
import asyncio
translator = Translator()

headers={
    'User-Agent' : ''
}

def getHtmlListScrollLoading(scrollTimes=5, scrollPauseTime=5):
    driver = webdriver.Chrome()
    url = 'https://wallstreetcn.com/news/global'
    driver.get(url)
    initialHeight = driver.execute_script('return document.body.scrollHeight')
    print("initial page length: " + str(initialHeight))
    for i in range(scrollTimes):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(scrollPauseTime) # wait for loading page
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
    times = [i.find('time').text for i in meta]
    ids = [i.split('/')[-1] for i in newsLinks]
    returnValue = {
        "links" : newsLinks,
        "titles" : titles,
        "authors" : authors,
        "times" : times,
        "abstracts" : abstracts,
        "ids": id
    }
    return returnValue


def getHtml(newsId):
    url = "https://api-one-wscn.awtmt.com/apiv1/content/articles/{}?extract=0"
    response = requests.get(url.format(newsId), headers=headers).json()
    abstract = response['data']['content_short']
    print(abstract)
    return abstract

async def translateText(text):
    translated = await translator.translate(text, dest='en')
    return translated.text

def translate(text):
    return asyncio.run(translateText(text))
    
if __name__ == '__main__':
    htmlContent = getHtmlListScrollLoading(1)
    returnValue = parseHtmlList(htmlContent)
    for i in returnValue['abstracts']:
        print(translateText(i))
