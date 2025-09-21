# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('popular')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger_eng') 
# pos_tag requied package

# pyrhon jieba-0.42.1/setup.py install (install jieba library)
# chinese tokenizer

import nltk
import re
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('popular')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng') 
nltk.download('vader_lexicon')
nltk.download('wordnet')

stopWords = set(nltk.corpus.stopwords) # including the negation words
negationWords = {
    'no', 'not', 'nor', 'never', 'none', 'noone', 'nothing', 'nowhere',
    'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}
lemmatizer = nltk.stem.WordNetLemmatizer()
sia = nltk.sentiment.SentimenyIntensityAnalyzer()

def textTokenized(text):
    text = re.sub('[^w]+', ' ', text) # get rid of all punctuations
    tokens = nltk.tokenize.word_tokenize(text.lower()) # tokenize the lowercase text
    tokensPos = nltk.pos_tag(tokens) 





tokenized = nltk.word_tokenize('hello! everyone!')
print(tokenized)
tags = nltk.pos_tag(tokenized)
print(tags)