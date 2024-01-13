# Importing the libraries
import pandas as pd
# ! pip install GetOldTweets3
import GetOldTweets3 as got3
import datetime


def scrape_tweets(name_of_coin, coin_symbol, timespan=60 * 24, nTweets=5000):
    '''
    Scrape tweets dealing with the coin name for the desired timespan.

    Parameters
    -----------
    name_of_coin: the name of the token
    coin_symbol: the symbol of the token
    timespan: the timespan in which the tweets are scraped (minutes)
    nTweets: the maximum number of tweets

    return
    -------
    Pandas dataframe containing the tweets information dealing with the token
    '''
    querywords = "{0} Crypto OR {0} Cryptocurrency OR {0} Coin OR {0} Token OR {1} Crypto OR {1} Cryptocurrency OR {1} Coin OR {1} Token OR ${1}".format(name_of_coin, coin_symbol)
    date = []
    username = []
    text = []
    hashtags = []
    retweets = []
    favorites = []
    mentions = []
    to = []

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    now_time = datetime.datetime.now()
    previous_time = now_time - datetime.timedelta(minutes=timespan)

    print(f'Time of first scraped tweet: {datetime.datetime.now()}')
    print('Starting scraping tweets...')

    tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(querywords).setMaxTweets(nTweets).setUntil(tomorrow.strftime("%Y-%m-%d")).setLang("en").setSince(previous_time.strftime("%Y-%m-%d"))
    get_tweet = got3.manager.TweetManager.getTweets(tweetCriteria)

    # Updating the lists that will serve to create the DF
    for tweet in get_tweet:
        date.append(tweet.date)
        username.append(tweet.username)
        text.append(tweet.text)
        hashtags.append(tweet.hashtags)
        retweets.append(tweet.retweets)
        favorites.append(tweet.favorites)
        mentions.append(tweet.mentions)
        to.append(tweet.to)

    tweets = pd.DataFrame({"date": date, "username": username, "text": text, "hashtags": hashtags, "mentions": mentions, "retweets": retweets, "favorites": favorites, "to": to})
    print('Done.')
    tweets = tweets[tweets.date > pd.to_datetime(previous_time).tz_localize('US/Eastern')]

    return tweets
