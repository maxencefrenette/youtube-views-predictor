import pandas as pd
import numpy as np

def process_data():
    
    # Read CSV file
    yt_data = pd.read_csv('../data/youtube_video_data.csv')

    # Drop the CSV file's numeric row count/index column
    yt_data = yt_data.drop('Unnamed: 0', axis=1)

    # Convert 2 date columns into datetime objects
    yt_data['video_published_at'] = pd.to_datetime(yt_data['video_published_at'])
    yt_data['channel_published_at'] = pd.to_datetime(yt_data['channel_published_at'])

    #Randomize the indices
    np.random.seed(0)
    yt_data = yt_data.reindex(np.random.permutation(yt_data.shape[0]))

    #drop dup video ids
    yt_data.drop_duplicates(subset=['video_id'], inplace=True)

    #data preprocessing

    # Filter to show columns with NaN values
    na_cols = yt_data.columns[yt_data.isna().any()]

    #fill missing likes based on like_to_view_ratio
    like_to_view_ratio = yt_data['likes'].sum() / yt_data['views'].sum()
    yt_data['likes'] = yt_data['likes'].fillna(yt_data['views'] * like_to_view_ratio)

    #fill missing comments based on comments_to_view_ratio
    comment_to_view_ratio = yt_data['comments'].sum() / yt_data['views'].sum()
    yt_data['comments'] = yt_data['comments'].fillna(yt_data['views'] * comment_to_view_ratio)

    #Feature engineer a few ratio fields
    yt_data['likes_to_views'] = yt_data['likes'] / yt_data['views']
    yt_data['comments_to_views'] = yt_data['comments'] / yt_data['views']

    # Filter to show columns with NaN values
    na_cols = yt_data.columns[yt_data.isna().any()]

    #log transform variables with a high skewness: 
    yt_data['log_views'] = np.log(yt_data['views'])
    yt_data['log_hours_published_video'] = np.log(yt_data['hours_published_video'])
    yt_data['log_likes'] = np.log(yt_data['likes'])
    yt_data['log_comments'] = np.log(yt_data['comments'])
    yt_data['log_channel_views'] = np.log(yt_data['channel_views'])
    yt_data['log_channel_subscribers'] = np.log(yt_data['channel_subscribers'])
    yt_data.drop(columns = ['views','hours_published_video','likes','comments','channel_views','channel_subscribers'], inplace=True)

    #likes, and comments are all outcome variables rather than predictors. 
    #drop these features. Likes / dislikes ratio was discussed but was not included in API data file. 
    yt_data.drop(columns= ['log_likes','log_comments'],inplace=True)

    yt_data = yt_data.astype({col: float for col in yt_data.select_dtypes(include='int64').columns})

    #thumbnail width, thumbnail height, and favorites only contain a single value each 
    #with no variability. Drop these features. 
    yt_data.drop(columns= ['thumbnail_height','thumbnail_width','favorites'], inplace= True)

    #drop channel_views since highly correlated with channel_subscribers
    yt_data.drop(columns=['log_channel_views'], inplace = True)

    # Split into training & test sets
    # 1,274 rows total -- approx 80/20 split
    train = yt_data[:1000]
    test = yt_data[1000:]

    X_train = train.drop(['log_views'], axis=1)
    y_train = train['log_views']
    X_test = test.drop(['log_views'], axis=1)
    y_test = test['log_views']
    
    return X_train, y_train,X_test,y_test

