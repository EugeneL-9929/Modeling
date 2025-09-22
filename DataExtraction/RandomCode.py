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