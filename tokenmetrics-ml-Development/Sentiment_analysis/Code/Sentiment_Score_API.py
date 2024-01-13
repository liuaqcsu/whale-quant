# Importing libraries
import pandas as pd
from Tools import twitterscraping, datapreprocessing, sentimentscoring
import json


# Input here the name of coin and its symbol:
####################################################
coin_name = 'Ethereum'
coin_symbol = "ETH"
####################################################

# Scraping tweets
data = twitterscraping.scrape_tweets(coin_name, coin_symbol, timespan=24 * 60)

# Preprocessing tweets
data = datapreprocessing.preprocess_twitter(data, scam_dict='Tools/scam_keywords.txt').copy()

# Extracting sentiment of tweets
sentiment_data = sentimentscoring.get_sentiment(data)

# Creating the API
nb_tweets = int(len(sentiment_data))
very_neg = len(sentiment_data[sentiment_data.sentiment == 'very negative']) / len(sentiment_data)
neg = len(sentiment_data[sentiment_data.sentiment == 'negative']) / len(sentiment_data)
neutral = len(sentiment_data[sentiment_data.sentiment == 'neutral']) / len(sentiment_data)
pos = len(sentiment_data[sentiment_data.sentiment == 'positive']) / len(sentiment_data)
very_pos = len(sentiment_data[sentiment_data.sentiment == 'very positive']) / len(sentiment_data)
avg_score = sentimentscoring.weighted_score(sentiment_data, sentiment_col='sentiment_score', weight_col='favorites')
index = ['coin name', 'coin symbol', 'number of tweets', 'very negative', 'negative', 'neutral', 'positive', 'very positive', 'average sentiment score']
sentiment_api = pd.Series([coin_name, coin_symbol, nb_tweets, very_neg, neg, neutral, pos, very_pos, avg_score], index).to_json()

with open(f'sentiment_api_{coin_symbol}.json', 'w') as outfile:
    json.dump(sentiment_api, outfile)
