import json
import numpy as np
import os
import pickle
import requests
import sys

from bs4 import BeautifulSoup

from network import Network
from settings import DATA_DIR


AUTOCOMPLETE_URL = 'http://www.fipiran.com/DataService/AutoCompleteindex'
EXPORT_URL = 'http://www.fipiran.com/DataService/Exportindex'
START_DATE = 13800101  # YYYYMMDD Solar Hijri calendar
END_DATE = 14000101  # YYYYMMDD Solar Hijri calendar
TIME_FRAMES = 100


class IranStock:
    def __init__(self):
        self._instrument_id_to_node_index = {}
        self._instrument_ids = []
        self._names = []
        self._get_stock_indices()
        self._date_to_time_frame_index = {}
        self._dates = []
        self._raw_data = None
        self._get_raw_data()
        self._fill_raw_data_empty_entries()
        self.networks = []
        self._create_networks()

    def _get_stock_indices(self):
        cached_path = os.path.join(DATA_DIR, 'iran_stock_indices.json')
        if os.path.exists(cached_path):
            with open(cached_path, 'r') as cached_file:
                response_json = json.loads(cached_file.read())
        else:
            response = requests.post(AUTOCOMPLETE_URL, data={
                'id': '',
            })
            if response.status_code == 200:
                with open(cached_path, 'w') as cached_file:
                    cached_file.write(response.text)
                response_json = json.loads(response.text)
            else:
                response_json = []

        counter = 0
        for item in response_json:
            name = item['LVal30']
            if name[0].isdigit():
                instrument_id = item['InstrumentID']
                self._instrument_id_to_node_index[instrument_id] = counter
                counter += 1
                self._instrument_ids.append(instrument_id)
                self._names.append(name)

    def _get_raw_data(self):
        dates = set()
        entries = []
        for i, instrument_id in enumerate(self._instrument_ids):
            # progress bar
            sys.stdout.write('\rExcel Files [%d/%d]' % (i + 1, len(self._instrument_ids)))
            sys.stdout.flush()

            cached_path = os.path.join(DATA_DIR, instrument_id)
            if os.path.exists(cached_path):
                with open(cached_path, 'r') as cached_file:
                    response_text = cached_file.read()
            else:
                response = requests.post(EXPORT_URL, data={
                    'inscodeindex': instrument_id,
                    'indexStart': START_DATE,
                    'indexEnd': END_DATE,
                })
                response_text = response.text
                with open(cached_path, 'w') as cached_file:
                    cached_file.write(response_text)

            soup = BeautifulSoup(response_text, features='html.parser')
            table = soup.find('table')
            if table:
                for row in table.findAll('tr'):
                    if row:
                        columns = row.findAll('td')
                        if columns:
                            date = columns[1].string.strip()
                            dates.add(date)
                            amount = columns[2].string.strip()
                            entries.append((date, instrument_id, amount))
        print()  # newline

        self._dates = sorted(dates)
        for i, date in enumerate(self._dates):
            self._date_to_time_frame_index[date] = i

        self._raw_data = np.zeros((len(self._dates), len(self._instrument_ids)))
        for entry in entries:
            date = entry[0]
            time_frame_index = self._date_to_time_frame_index[date]
            instrument_id = entry[1]
            node_index = self._instrument_id_to_node_index[instrument_id]
            amount = entry[2]
            self._raw_data[time_frame_index, node_index] = amount

    def _fill_raw_data_empty_entries(self):
        for column_index in range(self._raw_data.shape[1]):
            previous_value = 0
            for row_index in range(self._raw_data.shape[0]):
                if not self._raw_data[row_index, column_index]:
                    self._raw_data[row_index, column_index] = previous_value
                else:
                    previous_value = self._raw_data[row_index, column_index]

    def _create_networks(self):
        number_of_networks = int(2 * self._raw_data.shape[0] / TIME_FRAMES) + 1
        for i in range(number_of_networks):
            # progress bar
            sys.stdout.write('\rNetworks [%d/%d]' % (i + 1, number_of_networks))
            sys.stdout.flush()

            start_index = i * int(TIME_FRAMES / 2)
            end_index = start_index + TIME_FRAMES
            x = self._raw_data[start_index:end_index]
            time_frame_labels = self._dates[start_index:end_index]
            node_labels = self._names
            self.networks.append(Network(x, time_frame_labels, node_labels))
        print()  # newline


def get_iran_stock_networks(recreate=False):
    cached_path = os.path.join(DATA_DIR, 'iran_stock_networks.p')
    if not recreate and os.path.exists(cached_path):
        with open(cached_path, 'rb') as cached_file:
            networks = pickle.load(cached_file)
    else:
        networks = IranStock().networks
        with open(cached_path, 'wb') as cached_file:
            pickle.dump(networks, cached_file)
    return networks
