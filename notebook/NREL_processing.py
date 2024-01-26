import csv
import requests

API_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1.csv"
API_PARAMS = {'api_key': 'EUe0n9InavfhKtKtmscW1Emd5b3IhaJwOkcHu3MN', 'fuel_type': 'ELEC'}
RAW_FILE = "../data/NREL_raw.csv"

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


if __name__ == '__main__':
    get_data(API_URL, API_PARAMS, RAW_FILE)