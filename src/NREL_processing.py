import os
import re
import csv
import argparse
import requests
import numpy as np
import pandas as pd

API_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1.csv"
API_PARAMS = {
    'api_key': 'EUe0n9InavfhKtKtmscW1Emd5b3IhaJwOkcHu3MN', 'fuel_type': 'ELEC'}
RAW_FILE = "../data/NREL_raw.csv"
CORPUS_FILE = "../data/NREL_corpus.jsonl"
NUMERICAL_FILE = "../data/NREL_numerical.csv"

EV_columns = [
    'Station Name', 'Street Address', 'Intersection Directions',
    'City', 'State', 'ZIP', 'Plus4', 'Station Phone',
    'Access Days Time', 'Cards Accepted',
    'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count',
    'EV Other Info', 'EV Network', 'EV Network Web',
    'Geocode Status', 'Latitude', 'Longitude',
    'Date Last Confirmed', 'ID', 'Updated At', 'Owner Type Code',
    'Federal Agency ID', 'Federal Agency Name', 'Open Date',
    'EV Connector Types', 'Country', 'Access Code', 'Access Detail Code',
    'Federal Agency Code', 'Facility Type',
    'EV Pricing', 'EV On-Site Renewable Source', 'Restricted Access',
    'NPS Unit Name', 'Maximum Vehicle Class', 'EV Workplace Charging'
]

core_params = [
    'ID', 'Cards Accepted', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count',
    'EV Network', 'Owner Type Code',
    'EV Connector Types', 'Facility Type',
    'EV Pricing', 'EV On-Site Renewable Source',
    'Maximum Vehicle Class', 'EV Workplace Charging', 'Latitude', 'Longitude',
    'Status Code', 'Groups With Access Code', 'Access Days Time', 'Restricted Access',
    'Access Code', 'Access Detail Code'
]

str_columns = [
    'Street Address', 'Intersection Directions', 'City',
    'State', 'ZIP', 'Plus4', 'Station Phone', 'Access Days Time',
    'EV Other Info', 'EV Network', 'EV Network Web',
    'EV Connector Types', 'Country', 'Access Code', 'Access Detail Code',
    'Facility Type', 'EV Pricing', 'EV On-Site Renewable Source',
    'NPS Unit Name',
]

enum_columns = [
    'Cards Accepted', 'Owner Type Code',
    'Federal Agency Name', 'Federal Agency Code',
    'Maximum Vehicle Class'
]

num_columns = [
    'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count',
    'Latitude', 'Longitude', 'ID', 'Federal Agency ID'
]

vehicle_class_map = {
    'LD': 'Passenger vehicles (class 1-2)',
    'MD': 'Medium-duty (class 3-5)',
    'HD': 'Heavy-duty (class 6-8)'
}

owners_map = {
    'FG': 'Federal Government Owned',
    'J': 'Jointly Owned',
    'LG': 'Local/Municipal Government Owned',
    'P': 'Privately Owned',
    'SG': 'State/Provincial Government Owned',
    'T': 'Utility Owned'
}

cards_map = {
    'A': 'American Express',
    'C': 'Credit',
    'Debit': 'Debit',
    'D': 'Discover',
    'M': 'MasterCard',
    'V': 'Visa',
    'Cash': 'Cash',
    'Checks': 'Checks',
    'ACCOUNT_BALANCE': 'Account Balance',
    'ALLIANCE': 'Alliance AutoGas',
    'ANDROID_PAY': 'Android Pay',
    'APPLE_PAY': 'Apple Pay',
    'ARI': 'ARI',
    'CleanEnergy': 'Clean Energy',
    'Comdata': 'Comdata',
    'CFN': 'Commercial Fueling Network',
    'EFS': 'EFS',
    'FleetOne': 'Fleet One',
    'FuelMan': 'Fuelman',
    'GasCard': 'GASCARD',
    'PacificPride': 'Pacific Pride',
    'PHH': 'PHH',
    'Proprietor': 'Proprietor Fleet Card',
    'Speedway': 'Speedway',
    'SuperPass': 'SuperPass',
    'TCH': 'TCH',
    'Tchek': 'T-Chek T-Card',
    'Trillium': 'Trillium',
    'Voyager': 'Voyager',
    'Wright_Exp': 'WEX'
}

ev_networks_list = ['ChargePoint Network',
                    'Non-Networked',
                    'Blink Network',
                    'Tesla',
                    'Volta',
                    'EV Connect',
                    'SHELL_RECHARGE',
                    'eVgo Network',
                    'Electrify America',
                    'AMPUP',
                    'FLO',
                    'RIVIAN',
                    'LIVINGSTON',
                    ]


def get_data(url, params, save_path):
    r = requests.get(url, params=params)
    print('Status Code: ', r.status_code)
    decoded_content = r.content.decode('utf-8')
    reader = csv.reader(decoded_content.splitlines(), delimiter=',')
    with open(save_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in reader:
            writer.writerow(row)
    print('Data saved to: ', save_path)


def make_jsonl(save_path):
    raw_data.fillna('', inplace=True)
    # raw_data['Cards Accepted'].replace(cards_map, inplace=True)
    # raw_data['Owner Type Code'].replace(owners_map, inplace=True)
    # raw_data['Maximum Vehicle Class'].replace(vehicle_class_map, inplace=True)
    str_data = raw_data[str_columns + enum_columns]
    concatenated = str_data.apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)
    df_conc = pd.DataFrame(concatenated, columns=['text'])
    df_conc['docid'] = raw_data['ID'].astype(str)
    df_conc['name'] = raw_data['Station Name']
    df_conc = df_conc[['docid', 'name', 'text']]
    df_conc.to_json(save_path, orient='records', lines=True)


def get_access_score(row):
    score = 5
    if row['Status Code'] != 'E':
        return 0
    if 'public' in row['Access Code'].lower():
        if row['Groups With Access Code'].lower != 'public' or row['Access Detail Code'] != np.NAN:
            score -= 1
    else:
        return 1
    if row['Access Days Time'] != '24 hours daily':
        score -= 1
    return score


def get_park_price(x):
    x = str(x)
    if str(x).lower() == 'free':
        return 0
    # Extract hourly rate
    hourly_rate = re.findall(
        r'(\$\d+\.?\d*)-(\$\d+\.?\d*)/Hr \w+ Parking Fee', x)
    if not hourly_rate:
        hourly_rate = re.findall(r'(\$\d+\.?\d*)/Hr Parking Fee', x)
    if not hourly_rate:
        hourly_rate = re.findall(r'(\$\d+\.?\d*) per hour', x)
    if not hourly_rate:
        hourly_rate = re.findall(r'(\$\d+\.?\d*) per minute', x)
        hourly_rate = re.findall(r'\d+\.?\d*', str(hourly_rate))
        hourly_rate = [float(rate) * 60 for rate in hourly_rate]
    # Convert to float if hourly_rate is found else return None
    hourly_rate = re.findall(r'\d+\.?\d*', str(hourly_rate))
    for i in range(len(hourly_rate)):
        hourly_rate[i] = float(hourly_rate[i])
    if hourly_rate:
        return np.mean(hourly_rate)
    else:
        return 0


def get_electric_price(x):
    x = str(x)
    if x.lower() == 'free':
        return 0
    # Extract hourly rate
    rate = re.findall(r'(\$\d+\.?\d*)-(\$\d+\.?\d*)/kWh \w+ Energy Fee', x)
    if not rate:
        rate = re.findall(r'(\$\d+\.?\d*)/kWh Energy Fee', x)
    if not rate:
        rate = re.findall(r'(\$\d+\.?\d*) per kWh', x)
    # Convert to float if rate is found else return None
    rate = re.findall(r'\d+\.?\d*', str(rate))
    for i in range(len(rate)):
        rate[i] = float(rate[i])
    if rate:
        return np.mean(rate)
    else:
        return 0


def map(x):
    if x == 'LD':
        return 1
    elif x == 'MD':
        return 2
    elif x == 'HD':
        return 3
    else:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process NREL data and save it to a file')
    parser.add_argument('r', help='Refresh NREL data')
    args = parser.parse_args()

    print('Processing NREL data...')
    if not os.path.exists(RAW_FILE) or args.r:
        get_data(API_URL, API_PARAMS, RAW_FILE)
    raw_data = pd.read_csv(RAW_FILE, delimiter='\t')
    raw_data = raw_data[raw_data['Status Code'] == 'E']
    raw_data = raw_data[EV_columns]
    print('Printing Corpus...')
    make_jsonl(CORPUS_FILE)

    raw_data = pd.read_csv(RAW_FILE, delimiter='\t')
    raw_data = raw_data[core_params]
    raw_data['Access Score'] = raw_data.apply(get_access_score, axis=1)
    access_cols = ['Status Code', 'Groups With Access Code',
                   'Access Days Time', 'Restricted Access',
                   'Access Code', 'Access Detail Code']
    try:
        raw_data.drop(columns=access_cols, inplace=True)
    except KeyError:
        pass
    print('Processing numerical data...')
    num_attribs = raw_data.select_dtypes('number').columns.to_list()
    cat_attribs = raw_data.select_dtypes('object').columns.to_list()
    bool_attribs = raw_data.select_dtypes('bool').columns.to_list()
    raw_data['Park Pricing'] = raw_data['EV Pricing'].apply(get_park_price)
    raw_data['Electric Pricing'] = raw_data['EV Pricing'].apply(
        get_electric_price)
    raw_data.drop('EV Pricing', axis=1, inplace=True)
    raw_data['EV Level1 EVSE Num'].fillna(0, inplace=True)
    raw_data['EV Level2 EVSE Num'].fillna(0, inplace=True)
    raw_data['EV DC Fast Count'].fillna(0, inplace=True)
    raw_data['Cards Accepted Num'] = raw_data['Cards Accepted'].apply(
        lambda x: len(x.split(' ')) if type(x) == str else 0)
    raw_data.drop('Cards Accepted', axis=1, inplace=True)
    raw_data['EV Connector Types Num'] = raw_data['EV Connector Types'].apply(
        lambda x: len(x.split(' ')) if type(x) == str else 0)
    raw_data.drop('EV Connector Types', axis=1, inplace=True)
    raw_data['Network Enum'] = raw_data['EV Network'].apply(
        lambda x: ev_networks_list.index(x) + 1 if x in ev_networks_list else 0)
    raw_data.drop('EV Network', axis=1, inplace=True)
    owner_type_list = raw_data['Owner Type Code'].value_counts(
    ).index[:10].to_list()
    raw_data['Owner Type Code Enum'] = raw_data['Owner Type Code'].apply(
        lambda x: owner_type_list.index(x) + 1 if x in owner_type_list else 0)
    raw_data.drop('Owner Type Code', axis=1, inplace=True)
    raw_data['EV On-Site Renewable Source'] = raw_data['EV On-Site Renewable Source'].apply(
        lambda x: 1 if x == 'SOLAR' else 0)
    raw_data['Maximum Vehicle Class Enum'] = raw_data['Maximum Vehicle Class'].apply(
        map)
    raw_data.drop('Maximum Vehicle Class', axis=1, inplace=True)
    facilities = raw_data['Facility Type'].value_counts()[
        :10].index.to_list()
    raw_data['Facility Type Enum'] = raw_data['Facility Type'].apply(
        lambda x: facilities.index(x) + 1 if x in facilities else 0)
    raw_data.drop('Facility Type', axis=1, inplace=True)
    print('Saving numerical data...')
    raw_data.to_csv(NUMERICAL_FILE, index=False, encoding='utf-8')
