from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from textblob import TextBlob
import time as t
from configparser import ConfigParser
import pandas as pd
import numpy as np
# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (12, 10)})
# sns.set_style('white')
plt.style.use('fivethirtyeight')

# Plotting libraries
from tqdm import tqdm
tqdm.pandas()


# Building the sentiment analysis functions
def get_senti(score):
    '''
    Classify the sentiment category based on polarity score

    Parameters
    ----------
    score: polarity score

    Return
    -------
    Sentiment category (str)
    '''
    if score < -0.5:
        return 'very negative'
    elif score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    elif score < 0.5:
        return 'positive'
    else:
        return 'very positive'


def blob_sentiment(dataframe, col_text='text'):
    '''
    Compute the sentiment of all rows in a Dataframe using TextBlob.

    Parameters
    -----------
    dataframe: the dataframe containing textual information
    col_text: the name of the column which contains the text to be analyzed

    Return
    ----------
    A new dataframe with columns for the TextBlob sentiment
    '''
    print("Computing sentiment with Textblob...")
    df = dataframe.copy()
    t.sleep(1)
    df['blob_polarity'] = df[col_text].progress_apply(lambda x: TextBlob(x).sentiment.polarity)
    df['blob_sentiment'] = df.blob_polarity.apply(get_senti)
    return df


# Extracting sentiment with Vader
def vader_sentiment(dataframe, col_text='text'):
    '''
    Compute the sentiment of all rows in a Dataframe using Vader.

    Parameters
    -----------
    dataframe: the dataframe containing textual information
    col_text: the name of the column which contains the text to be analyzed

    Return
    ----------
    A new dataframe with columns for the Vader sentiment
    '''
    print("Computing sentiment with Vader...")
    df = dataframe.copy()
    t.sleep(1)
    df['vader_polarity'] = df[col_text].progress_apply(lambda x: analyser.polarity_scores(x)['compound'])
    df['vader_sentiment'] = df.vader_polarity.apply(get_senti)
    return df


def get_sentiment(dataframe, col_text='text'):
    '''
    Compute the sentiment of all rows in a Dataframe using an average of Tetblob and Vader.

    Parameters
    -----------
    dataframe: the dataframe containing textual information
    col_text: the name of the column which contains the text to be analyzed

    Return
    ----------
    A new dataframe with columns for the computed sentiment scores
    '''
    df = dataframe.copy()
    df = blob_sentiment(df)
    df = vader_sentiment(df)
    df['sentiment_score'] = (df.blob_polarity + df.vader_polarity) / 2
    df['sentiment'] = df.sentiment_score.apply(get_senti)
    print('Done.')
    return df


def print_worse(sentiment_data, n=5, sentiment_col='sentiment_score'):
    '''
    print the top n worst records of the dataframe in terms of sentiment.

    Parameters
    ------------
    sentiment_data: the dataframe containing text and pre-computed sentiment scores
    n: the number of records to show
    sentiment_col: the column in the dataframe which contains the pre-computed sentiment scores
    '''
    for idx, txt in enumerate(sentiment_data.sort_values(sentiment_col, ascending=True).head(n).text):
        print(str(idx + 1) + ') ' + txt + '\n')


def print_best(sentiment_data, n=5, sentiment_col='sentiment_score'):
    '''
    print the top n best records of the dataframe in terms of sentiment.

    Parameters
    ------------
    sentiment_data: the dataframe containing text and pre-computed sentiment scores
    n: the number of records to show
    sentiment_col: the column in the dataframe which contains the pre-computed sentiment scores
    '''
    for idx, txt in enumerate(sentiment_data.sort_values(sentiment_col, ascending=False).head(n).text):
        print(str(idx + 1) + ') ' + txt + '\n')


def weighted_score(sentiment_data, sentiment_col, weight_col):
    '''
    compute the weighted average of sentient using some sort of weights, like the number of favorites for twitter data

    Parameters
    ------------
    sentiment_data:  the dataframe containing text and pre-computed sentiment scores
    sentiment_col: the column in the dataframe which contains the pre-computed sentiment scores
    weight_col: the column in the dataframe which contains the pre-computed weights

    Return
    ---------
    The weigthed average sentiment score for the whole dataset

    '''
    score_list = sentiment_data[sentiment_col]
    weight_list = sentiment_data[weight_col]
    weight_list = (weight_list + 1) / (len(sentiment_data) + weight_list.sum())
    return (score_list * weight_list).sum() / weight_list.sum()


def get_price_data(coin_symbol, config_file='/Users/pauldoan/Documents/Token Metrics/1 Admin/sql_crendentials.ini'):
    '''
    Retrieve the price data corresponding to coin_name from the Token Metrics database.

    Parameters
    -----------
    coin_symbol: the symbol of the token
    config_file: ini file which contains Token Metrics db credentials

    return
    -------
    The data frame containing all the price data for the coin.
    '''
    config = ConfigParser()
    config.read(config_file)
    user = config['SQL']['user']
    password = config['SQL']['password']

    import mysql.connector
    token_metrics_db = mysql.connector.connect(user=user, password=password,
                                               host='tokenmetrics-restored-27-05.cxuzrhvtziar.us-east-1.rds.amazonaws.com',
                                               database='tokenmetrics')
    crypto_prices = pd.read_sql_query("SELECT * FROM ico_price_daily_summaries WHERE currency = 'USD'", token_metrics_db)
    token_metrics_db.close()
    prices = crypto_prices[crypto_prices['ico_symbol'] == coin_symbol].copy()
    prices.loc[:, 'date'] = pd.to_datetime(prices.date)
    if len(prices) == 0:
        print('No price data for this coin.')
        return None
    print('Price data retrieved.')
    return prices


def compute_simple_moving_average(sentiment_data, window1=12, window2=30, twitter=False):

    # Weighted average group by
    def weigthed_average(data, quantity, weights):
        try:
            return (data[quantity] * data[weights]).sum() / data[weights].sum()
        except ZeroDivisionError:
            return data[quantity]

    if twitter:
        group_day = sentiment_data.groupby('day').apply(weigthed_average, 'sentiment_score', 'favorites').to_frame(name='sentiment_score')
    else:
        group_day = sentiment_data.groupby('day')['sentiment_score'].apply(np.average).to_frame(name='sentiment_score')

    group_day[f'moving_average_{window1}'] = group_day.iloc[:, 0].rolling(window=window1).mean()
    group_day[f'moving_average_{window2}'] = group_day.iloc[:, 0].rolling(window=window2).mean()
    group_day[f'velocity_{window2}'] = np.abs(group_day[f'moving_average_{window2}'] - group_day[f'moving_average_{window2}'].shift())

    return group_day


def plot_simple_moving_average(sentiment_data, price_data=None, window1=12, window2=30, twitter=False, figsize=(15, 10)):

    group_day = compute_simple_moving_average(sentiment_data, window1, window2, twitter)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [5, 3]})
    ax[0].set_title("Sentiment Moving Average", color='#024515')
    ax[0].tick_params(axis='y', labelcolor='#024515')
    ax[0].tick_params(axis='x', labelcolor='#024515')
    ax[0].set_ylabel('Sentiment', color='#024515')
    ax[0].set_xlabel('time', color='#024515')

    if price_data is not None:
        maskinf = np.max((sentiment_data.date.apply(lambda x: x.date()).min(), price_data.date.apply(lambda x: x.date()).min()))
        masksup = np.min((sentiment_data.date.apply(lambda x: x.date()).max(), price_data.date.apply(lambda x: x.date()).max()))
        sentiment_mask = (group_day.index >= maskinf) & (group_day.index <= masksup)
        prices_mask = (price_data.date >= pd.to_datetime(maskinf)) & (price_data.date <= pd.to_datetime(masksup))

        sns.lineplot(group_day.index, group_day[sentiment_mask][f'moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[sentiment_mask][f'moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

        ax2 = ax[0].twinx()
        ax2.set_ylabel('Price', color='#18288f')
        sns.lineplot(price_data[prices_mask].date.values, price_data[prices_mask].close.values, ax=ax2, color='#18288f', label='Price', linewidth=0.8)
        ax2.legend(loc='upper right')

    else:
        sns.lineplot(group_day.index, group_day[f'moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[f'moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

    ax[1].set_title('Velocity', color='#5e5e5e')
    ax[1].set_ylabel('Velocity', color='#5e5e5e')
    ax[1].tick_params(axis='y', labelcolor='#5e5e5e')
    sns.lineplot(group_day.index, group_day[f'velocity_{window2}'].values, color='#5e5e5e', ax=ax[1], label=f'Velocity - {window2} days', linewidth=0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def compute_weighted_moving_average(sentiment_data, window1=12, window2=30, twitter=False):
    weigth1 = np.arange(1, window1 + 1)
    weigth2 = np.arange(1, window2 + 1)

    # Weighted average group by
    def weigthed_average(data, quantity, weights):
        try:
            return (data[quantity] * data[weights]).sum() / data[weights].sum()
        except ZeroDivisionError:
            return data[quantity]

    if twitter:
        group_day = sentiment_data.groupby('day').apply(weigthed_average, 'sentiment_score', 'favorites').to_frame(name='sentiment_score')
    else:
        group_day = sentiment_data.groupby('day')['sentiment_score'].apply(np.average).to_frame(name='sentiment_score')

    group_day[f'weigthed_moving_average_{window1}'] = group_day.iloc[:, 0].rolling(window=window1).apply(lambda x: np.dot(x, weigth1) / weigth1.sum(), raw=True)
    group_day[f'weigthed_moving_average_{window2}'] = group_day.iloc[:, 0].rolling(window=window2).apply(lambda x: np.dot(x, weigth2) / weigth2.sum(), raw=True)
    group_day[f'weigthed_velocity_{window2}'] = np.abs(group_day[f'weigthed_moving_average_{window2}'] - group_day[f'weigthed_moving_average_{window2}'].shift())

    return group_day


def plot_weighted_moving_average(sentiment_data, price_data=None, window1=12, window2=30, twitter=False, figsize=(15, 10)):

    group_day = compute_weighted_moving_average(sentiment_data, window1, window2, twitter)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [5, 3]})
    ax[0].set_title("Sentiment Weighted Moving Average", color='#024515')
    ax[0].tick_params(axis='y', labelcolor='#024515')
    ax[0].tick_params(axis='x', labelcolor='#024515')
    ax[0].set_ylabel('Sentiment', color='#024515')
    ax[0].set_xlabel('time', color='#024515')

    if price_data is not None:
        maskinf = np.max((sentiment_data.date.apply(lambda x: x.date()).min(), price_data.date.apply(lambda x: x.date()).min()))
        masksup = np.min((sentiment_data.date.apply(lambda x: x.date()).max(), price_data.date.apply(lambda x: x.date()).max()))
        sentiment_mask = (group_day.index >= maskinf) & (group_day.index <= masksup)
        prices_mask = (price_data.date >= pd.to_datetime(maskinf)) & (price_data.date <= pd.to_datetime(masksup))

        sns.lineplot(group_day.index, group_day[sentiment_mask][f'weigthed_moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[sentiment_mask][f'weigthed_moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

        ax2 = ax[0].twinx()
        ax2.set_ylabel('Price', color='#18288f')
        sns.lineplot(price_data[prices_mask].date.values, price_data[prices_mask].close.values, ax=ax2, color='#18288f', label='Price', linewidth=0.8)
        ax2.legend(loc='upper right')

    else:
        sns.lineplot(group_day.index, group_day[f'weigthed_moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[f'weigthed_moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

    ax[1].set_title('Velocity', color='#5e5e5e')
    ax[1].set_ylabel('Velocity', color='#5e5e5e')
    ax[1].tick_params(axis='y', labelcolor='#5e5e5e')
    sns.lineplot(group_day.index, group_day[f'weigthed_velocity_{window2}'].values, color='#5e5e5e', ax=ax[1], label=f'Velocity - {window2} days', linewidth=0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def compute_exp_moving_average(sentiment_data, window1=12, window2=30, twitter=False):

    # Weighted average group by
    def weigthed_average(data, quantity, weights):
        try:
            return (data[quantity] * data[weights]).sum() / data[weights].sum()
        except ZeroDivisionError:
            return data[quantity]

    group_day = compute_simple_moving_average(sentiment_data, window1, window2, twitter=twitter)

    mod_score1 = group_day.iloc[:, 0].copy()
    mod_score1.iloc[0: window1] = group_day[f'moving_average_{window1}'][0: window1]
    mod_score2 = group_day.iloc[:, 0].copy()
    mod_score2.iloc[0: window2] = group_day[f'moving_average_{window2}'][0: window2]

    group_day[f'exp_moving_average_{window1}'] = mod_score1.ewm(span=window1, adjust=False).mean()
    group_day[f'exp_moving_average_{window2}'] = mod_score2.ewm(span=window2, adjust=False).mean()
    group_day[f'exp_velocity_{window2}'] = np.abs(group_day[f'exp_moving_average_{window2}'] - group_day[f'exp_moving_average_{window2}'].shift())
    group_day.drop([f'moving_average_{window1}', f'moving_average_{window2}', f'velocity_{window2}'], axis=1, inplace=True)

    return group_day


def plot_exp_moving_average(sentiment_data, price_data=None, window1=12, window2=30, twitter=False, figsize=(15, 10)):

    group_day = compute_exp_moving_average(sentiment_data, window1, window2, twitter)

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [5, 3]})
    ax[0].set_title("Sentiment Weighted Moving Average", color='#024515')
    ax[0].tick_params(axis='y', labelcolor='#024515')
    ax[0].tick_params(axis='x', labelcolor='#024515')
    ax[0].set_ylabel('Sentiment', color='#024515')
    ax[0].set_xlabel('time', color='#024515')

    if price_data is not None:
        maskinf = np.max((sentiment_data.date.apply(lambda x: x.date()).min(), price_data.date.apply(lambda x: x.date()).min()))
        masksup = np.min((sentiment_data.date.apply(lambda x: x.date()).max(), price_data.date.apply(lambda x: x.date()).max()))
        sentiment_mask = (group_day.index >= maskinf) & (group_day.index <= masksup)
        prices_mask = (price_data.date >= pd.to_datetime(maskinf)) & (price_data.date <= pd.to_datetime(masksup))

        sns.lineplot(group_day.index, group_day[sentiment_mask][f'exp_moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[sentiment_mask][f'exp_moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

        ax2 = ax[0].twinx()
        ax2.set_ylabel('Price', color='#18288f')
        sns.lineplot(price_data[prices_mask].date.values, price_data[prices_mask].close.values, ax=ax2, color='#18288f', label='Price', linewidth=0.8)
        ax2.legend(loc='upper right')

    else:
        sns.lineplot(group_day.index, group_day[f'exp_moving_average_{window1}'].values, color='#5fb387', ax=ax[0], label=f'Sentiment - {window1} days', linewidth=0.8)
        sns.lineplot(group_day.index, group_day[f'exp_moving_average_{window2}'].values, color='#024515', ax=ax[0], label=f'Sentiment - {window2} days', linewidth=0.8)
        ax[0].legend(loc='upper left')

    ax[1].set_title('Velocity', color='#5e5e5e')
    ax[1].set_ylabel('Velocity', color='#5e5e5e')
    ax[1].tick_params(axis='y', labelcolor='#5e5e5e')
    sns.lineplot(group_day.index, group_day[f'exp_velocity_{window2}'].values, color='#5e5e5e', ax=ax[1], label=f'Velocity - {window2} days', linewidth=0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
