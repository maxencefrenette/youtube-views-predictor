import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_data(buckets=4):
    
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

    # Replace missing titles & descriptions with empty str
    yt_data['title'] = yt_data['title'].fillna('')
    yt_data['description'] = yt_data['description'].fillna('')
    
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
    
    
    #create view bucket features
    def create_categorical_variable(df, num_col,buckets):
        # Compute the percentile cutoffs
        percentile_cutoffs = list(range(0, 101, int(100/buckets)))
        percentile_values = [np.percentile(df[num_col], i) for i in percentile_cutoffs]

        # Create the categorical variable
        field_name = 'views_category_' + str(buckets)
        df[field_name] = pd.cut(df[num_col], bins=percentile_values, labels=percentile_cutoffs[:-1])
        df[field_name] = df[field_name].fillna(0)

        return df
    
    yt_data = create_categorical_variable(yt_data,'views',buckets=buckets)

    yt_data.drop(columns = ['log_views','views','hours_published_video','likes','comments','channel_views','channel_subscribers'], inplace=True)

    #likes, and comments are all outcome variables rather than predictors. 
    yt_data.drop(columns= ['log_likes','log_comments'],inplace=True)

    yt_data = yt_data.astype({col: float for col in yt_data.select_dtypes(include='int64').columns})

    #thumbnail width, thumbnail height, and favorites only contain a single value each 
    #with no variability. Drop these features. 
    yt_data.drop(columns= ['thumbnail_height','thumbnail_width','favorites'], inplace= True)

    #drop channel_views since highly correlated with channel_subscribers
    yt_data.drop(columns=['log_channel_views'], inplace = True)

   
    # Split into training & test sets
    y_label = 'views_category_'+str(buckets)
    X = yt_data.drop(columns=[y_label])
    y = yt_data[[y_label]]
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train,X_test,y_test

