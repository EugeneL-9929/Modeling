import nltk
import re

class TextTokenizer():
    def __init__(self, text):
        self.text = text
    
    def punctuationRemover(self):
        self.text = re.sub('[^w]+', '', self.text)