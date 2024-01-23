import pandas as pd
from tqdm import tqdm

DATA_PATH = '../data/'
CACHE_PATH = '../cache/'


def merge_lat_lng(df1 : pd.DataFrame, df2 : pd.DataFrame, on : str = None) -> pd.DataFrame :
    '''
    merge dataframes based on lat/lng, in order to align with the 'docid' column

    Note: this function will automatically rename the columns 'L/latitude' and 'L/longitude' to 'lat' and 'lng' respectively

    Args:
        df1 : a pandas dataframe includes 'lat', 'lng', ('docid') columns
        df2 : a pandas dataframe includes 'lat', 'lng', ('docid') columns
        on : 'first' or 'second' or 'None'

    Returns:
        a merged dataframe
    '''

    if on is not None and on not in ['first', 'second']:
        raise Exception('on must be "first" or "second" or None')

    rename_dict = {
        'Latitude': 'lat',
        'Longitude': 'lng',
        'latitude': 'lat',
        'longitude': 'lng',
    }

    df1.rename(columns=rename_dict, inplace=True)
    df2.rename(columns=rename_dict, inplace=True)

    if on is None and 'docid' not in df1.columns and 'docid' not in df2.columns:
        raise Exception('docid column is required in at least one dataframe')

    if on is None and 'docid' in df1.columns and 'docid' in df2.columns:
        raise Exception('docid column is required in only one dataframe, or specify on="first" or on="second"')

    if (on == 'first' and 'docid' in df1.columns) or (on is None and 'docid' in df1.columns):
        first_df = df1
        second_df = df2
    elif (on == 'second' and 'docid' in df2.columns) or (on is None and 'docid' in df2.columns):
        first_df = df2
        second_df = df1

    if first_df is None or second_df is None:
        raise Exception('the specified dataframe does not have docid columns')

    aligned_list = []
    del_list = []
    for i in tqdm(range(len(first_df))):
        mask = (abs(second_df.lat - first_df.lat[i]) < 0.001) & (abs(second_df.lng - first_df.lng[i]) < 0.001)
        masked_second = second_df[mask]
        if len(masked_second) > 0:
            aligned_list.append(masked_second.index[0])
        else:
            del_list.append(i)
    first_df = first_df.drop(del_list)
    first_df = first_df.reset_index(drop=True)
    filtered_second = second_df.iloc[aligned_list]
    filtered_second = filtered_second.reset_index(drop=True)
    assert len(filtered_second) == len(first_df)

    filtered_second = filtered_second.drop(['lat', 'lng'], axis=1)
    merged_df = pd.concat([first_df, filtered_second], axis=1)
    return merged_df


def merge_lat_lng_tester():
    df1 = pd.read_csv('data/relevance.train.csv')
    df2 = pd.read_csv('data/Google_Map_review_data_aggregated.csv')
    merged = merge_lat_lng(df1, df2, on='first')
    df1.dropna()
    print(merged.head())
    

def get_NREL_map(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath, delimiter='\t')
    df['index'] = df.index
    df = df [['index', 'Latitude', 'Longitude']]
    df.rename(columns = {
        'index' : 'docid', 
        'Latitude' : 'lat',
        'Longitude' : 'lng'
    }, inplace=True)

    df.to_csv('data/docid_NREL_map.csv', index=False)
    return df


if __name__ == '__main__':
    get_NREL_map('data/NREL_All_Stations_data.csv')