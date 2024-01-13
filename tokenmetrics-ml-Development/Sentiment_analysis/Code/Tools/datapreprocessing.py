import pandas as pd
import re
from tqdm import tqdm
import string
import time
tqdm.pandas()


# Cleaning text helper function
def text_cleaning(text):
    '''
    Clean the text by removing unwanted characters and noise.

    Parameters
    ----------
    text: the text to be cleaned

    return
    ----------
    Cleaned text
    '''
    # checking if text empty
    if (not re.search('[a-zA-Z0-9]', text)) or (type(text) != str) or (text == 'nan'):
        return ''
    text = re.sub(r'@', '', text)             # Remove @ mentions
    text = re.sub(r'#', '', text)              # Remove Hastags symbols
    text = re.sub(r'RT[\s]+', '', text)        # Remove RT mention
    text = re.sub(r'\n', ' ', text)              # Remove line terminator character
    text = re.sub(r'(pictwitter)\w+', '', text)     # Removing picture names
    text = re.sub(r'\xa0', ' ', text)               # Removing non breaking space character
    return text


def url_detect(text):
    '''
    Detect the number of url links in the text

    Parameters
    ----------
    text: the text in which url should be detected

    return
    ---------
    number of url links in the text
    '''
    list_url = re.findall(r'https?:\/\/.*', text)
    return len(list_url)


def url_only(text):
    '''
    Detect if the text is an url only (does not contain anything but an URL link)

    Parameters
    -----------
    text: the text to be evaluated

    return
    ---------
    True or false
    '''
    if re.search(r'^https?:\/\/.*$', text):
        return True
    return False


def preprocess_twitter(data, likes_threshold=5, retweets_threshold=5, scam_dict='Scam_keywords.txt'):
    '''
    Clean the twitter data by removing scams and preparing proper text for sentiment analysis.
    The dataframe must have the following column names:
    - date
    - username
    - text
    - hashtags
    - mentions
    - retweets
    - favorites
    Any other column will be deleted.

    Parameters
    -------------
    data: the dataframe containing tweets.
    likes_threshold: the number of likes threshold below which url tweet are considered as ad
    retweets_threshold: the number of retweets threshold below which url tweet are considered as ad
    scam_dict: the text file used to target scam keywords

    return
    ---------
    Cleaned twitter pandas Dataframe

    '''
    data.loc[:, 'date'] = pd.to_datetime(data.date)
    data = data[['username', 'date', 'hashtags', 'mentions', 'retweets', 'favorites', 'text']].copy()
    real_data = data.dropna(subset=['text']).copy()

    print('Detecting scam tweets...')
    time.sleep(1)
    real_data.loc[:, 'url'] = real_data.text.progress_apply(url_detect)

    with open(scam_dict, 'r') as file:
        content = file.readlines()
    scam_dictionary = [re.sub('\n', '', x.strip().lower()) for x in content]

    # Detect spam helper function
    def scam_detect(text):
        if any([re.search(f'[{string.punctuation} ]{word}[{string.punctuation} ]', text.lower()) for word in scam_dictionary]):
            return True
        if not re.search(' [a-zA-Z]+ ', text):
            return True

    mask1 = real_data.url > 0
    mask2 = real_data.favorites <= likes_threshold
    mask3 = real_data.retweets <= retweets_threshold
    mask4 = real_data.text.progress_apply(scam_detect)
    ads_data = real_data[mask1 & mask2 & mask3 | mask4].drop('url', axis=1)
    no_ads_data = real_data.drop(labels=ads_data.index).drop('url', axis=1)
    try:
        print(f"Remaining data: {round(100 * len(no_ads_data) / len(data))} %\n")
    except:
        pass

    print('Cleaning text...')
    time.sleep(1)
    no_ads_data.loc[:, 'text'] = no_ads_data.text.progress_apply(text_cleaning)

    no_ads_data.loc[:, 'day'] = no_ads_data.date.apply(lambda x: x.date())
    no_ads_data = no_ads_data[['username', 'date', 'day', 'hashtags', 'mentions', 'retweets', 'favorites', 'text']]
    print('Done.')

    return no_ads_data


def preprocess_telegram(data):
    '''
    Clean the telegram data by removing url only messages and preparing proper text for sentiment analysis.
    The dataframe must have the following column names:
    - message id
    - author
    - date
    - message
    Any other column will be deleted.

    Parameters
    -------------
    data: the dataframe containing telegram messages.
    scam_dict: the text file used to target scam keywords

    return
    ---------
    Cleaned telegram pandas Dataframe

    '''
    data.loc[:, 'date'] = pd.to_datetime(data.date)
    real_data = data.dropna(subset=['message']).copy()

    print('Detecting url only messages...')
    time.sleep(1)
    real_data.loc[:, 'url_only'] = real_data.message.progress_apply(url_only)

    mask1 = real_data.url_only == True
    ads_data = real_data[mask1].drop('url_only', axis=1)
    no_ads_data = real_data.drop(labels=ads_data.index).drop('url_only', axis=1)

    print('Cleaning text...')
    time.sleep(1)
    no_ads_data.loc[:, 'text'] = no_ads_data.message.progress_apply(text_cleaning)
    no_ads_data = no_ads_data.drop('message', axis=1)
    no_ads_data.loc[:, 'day'] = no_ads_data.date.apply(lambda x: x.date())
    no_ads_data = no_ads_data[no_ads_data.text != '']

    try:
        print(f"Remaining data: {round(100 * len(no_ads_data) / len(data))} %\n")
    except:
        pass
    print('Done.')

    return no_ads_data
