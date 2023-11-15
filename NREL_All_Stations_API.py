import csv
import requests

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