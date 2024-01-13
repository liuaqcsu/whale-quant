# Importing libraries
import pandas as pd
from configparser import ConfigParser
import nest_asyncio
import mysql.connector
nest_asyncio.apply()


def find_tm_group(coin_name, config_file='/Users/pauldoan/Documents/Token Metrics/1 Admin/sql_crendentials.ini'):
    '''
    Retrieve the telegram group ID corresponding to coin_name from the Token Metrics database.

    Parameters
    -----------
    coin_name: the name of the token
    config_file: ini file which contains Token Metrics db credentials

    return
    -------
    The telegram group ID
    '''
    config = ConfigParser()

    # Importing telegram info from Token Metrics database
    config.read(config_file)
    user = config['SQL']['user']
    password = config['SQL']['password']
    token_metrics_db = mysql.connector.connect(user=user, password=password,
                                               host='tokenmetrics-restored-27-05.cxuzrhvtziar.us-east-1.rds.amazonaws.com',
                                               database='tokenmetrics')
    telegram_link_full = pd.read_sql_query("SELECT id, name, telegram_group_name FROM icos order by id desc;", token_metrics_db)
    token_metrics_db.close()
    telegram_link = telegram_link_full[telegram_link_full.telegram_group_name.notna()]

    # Extracting information on token metrics db
    try:
        telegram_channel = telegram_link.loc[telegram_link.name == coin_name, 'telegram_group_name'].values[0]
        print('Telegram group found:', telegram_channel)
        return(telegram_channel)
    except:
        print("Check spelling of coin, or telegram group not in database...")
        return None


async def scrape_telegram_members(group_id, clt):
    '''
    Scrape information on the members of the Telegram group.

    Parameters
    -----------
    group_id: the ID of the telegram group
    clt: the connected telegram client

    return
    -------
    Pandas dataframe containing information on the members of the Telegram group
    '''

    # Import libraries
    from telethon.tl.functions.channels import GetParticipantsRequest
    from telethon.tl.types import ChannelParticipantsSearch
    from telethon.tl.types import PeerChannel

    # Extract telegram group id
    entity_id = group_id
    if entity_id.isdigit():
        entity = PeerChannel(int(entity_id))
    else:
        entity = entity_id

    my_channel = await clt.get_entity(entity)
    print(f"Getting members data for telegram group: {group_id} ...")

    # Scrape data on users of the telegram group
    offset = 0
    limit = 100
    all_participants = []
    counter = 0
    while True:
        participants = await clt(GetParticipantsRequest(my_channel, ChannelParticipantsSearch(''), offset, limit, hash=0))

        # If no more participants to scrape
        if not participants.users:
            print('No more participants to scrape.\nDone.\n')
            break

        all_participants.extend(participants.users)
        offset += len(participants.users)
        counter += 1
        if counter % 20 == 0:
            print(f'Scraped {len(all_participants)} users...')

    # Create Dataframe with the data
    all_user_details = []
    for participant in all_participants:
        all_user_details.append(
            {"id": participant.id, "first_name": participant.first_name, "last_name": participant.last_name,
             "user": participant.username, "phone": participant.phone, "is_bot": participant.bot})

    return pd.DataFrame(all_user_details)


async def scrape_telegram_messages(group_id, clt, limit_msgs=10000):
    '''
    Scrape the messages of the Telegram group.

    Parameters
    -----------
    group_id: the ID of the telegram group
    clt: the connected telegram client
    limit_msgs: the desired number of messages

    return
    -------
    Pandas dataframe containing messages of the Telegram group
    '''
    # Importing libraries
    from telethon.tl.functions.messages import GetHistoryRequest
    from telethon.tl.types import PeerChannel

    # Extract telegram group id
    entity_id = group_id
    if entity_id.isdigit():
        entity = PeerChannel(int(entity_id))
    else:
        entity = entity_id
    my_channel = await clt.get_entity(entity)
    print(f"Getting messages data for telegram group: {group_id} ...")

    # Scrape messages of the telegram group
    offset_id = 0
    limit = 100
    all_messages = []
    total_messages = 0
    total_count_limit = limit_msgs

    # Verbose for user
    counter = 0
    if limit_msgs < 5000:
        thresh = 4
    else:
        thresh = 10

    while True:
        if counter % thresh == 0 and counter != 0:
            print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
        history = await clt(GetHistoryRequest(
            peer=my_channel,
            offset_id=offset_id,
            offset_date=None,
            add_offset=0,
            limit=limit,
            max_id=0,
            min_id=0,
            hash=0
            ))

        # If no more messages to scrape
        if not history.messages:
            print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
            print('No more messages.\nDone.\n')
            break

        # Storing messages
        messages = history.messages
        for message in messages:
            all_messages.append(message.to_dict())
        offset_id = messages[len(messages) - 1].id
        total_messages = len(all_messages)
        counter += 1
        # Stop if more than limit
        if total_count_limit != 0 and total_messages >= total_count_limit:
            print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
            print('Limit reached.\nDone.\n')
            break

    # Create Dataframe with data
    return pd.DataFrame(all_messages)[['id', 'post_author', 'date', 'message']]
