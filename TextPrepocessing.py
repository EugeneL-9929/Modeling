import re
import urllib.request as request
import urllib.error as error
import urllib.parse as parse
import ssl
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

# ignore ssl certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def htmlPurify(html):
    clearHtmlTags = re.sub(r'<[^>]+>', '', html)
    clearHtmlTags = re.sub(r'\s+', ' ', clearHtmlTags)
    # print(clearHtmlTags)
    return clearHtmlTags

def htmlCreep(html, url):
    anchorLink = re.findall(r'<a\s+href="([^"^#]+)"', html)
    anchorLink = relativeUrlAbsolute(anchorLink)
    imgLink = re.findall(r'<img\s+src="([^"]+)"', html)
    imgLink = relativeUrlAbsolute(imgLink)
    # print(imgLink)
    return anchorLink, imgLink

def htmlClassId(html):
    classList = re.findall(r'class="([^"]+)"', html)
    idList = re.findall(r'id="([^"]+)"', html)
    print(classList)
    return classList, idList


def relativeUrlAbsolute(linkList):
    length = len(linkList)
    for i in range(length):
        linkList[i] = re.sub(r'\s', r'%20', linkList[i])
        if linkList[i].startswith(r'/'):
            linkList[i] = url + linkList[i]
    return linkList



if __name__ == '__main__':
    url = "https://hkust.edu.hk/"
    html = request.urlopen(url, context=ctx).read().decode()
    htmlClassId(html)
    # htmlCreep(html, url)
    # html = htmlPurify(html)
    # word = word_tokenize(html)
    # print(word)
