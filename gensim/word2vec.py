import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

print('over')