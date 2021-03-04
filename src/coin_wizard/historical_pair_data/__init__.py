#!/usr/bin/python3
import csv, os, pytz
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime
from coin_wizard.historical_pair_data_fetcher import download_hist_data
from coin_wizard.historical_pair_data_fetcher.api import Platform, TimeFrame
from zipfile import ZipFile
from numpy import genfromtxt

temp_directory = './temp'
platform = Platform.GENERIC_ASCII
time_frame = TimeFrame.ONE_MINUTE
pair_data_directory = 'pair_data/'
eastern = pytz.timezone('US/Eastern')
utc = pytz.utc

def remove_directory_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def update_a_historical_pair_data(output_directory, year="2016", month=None, pair="eurusd", download_again=False):
    # Determine filename
    if month is None:
        output_filename = 'DAT_{}_{}_{}_{}.npy'.format(platform, pair.upper(), time_frame, str(year))
    else:
        output_filename = 'DAT_{}_{}_{}_{}.npy'.format(platform, pair.upper(), time_frame, '{}{}'.format(year, str(month).zfill(2)))

    output_file_path = os.path.join(output_directory, output_filename)

    # Prevent re-download
    if os.path.exists(output_file_path) and download_again == False:
        return output_file_path

    # Makedir clear temp files
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    remove_directory_files(temp_directory)

    temp_filename = download_hist_data(year=year,month=month,pair=pair, output_directory=temp_directory, platform=platform, time_frame=time_frame, verbose=False)

    # Unzip temp files
    with ZipFile(temp_filename, 'r') as zipObj:
        # Extract all the contents of zip file in temp directory
        zipObj.extractall(temp_directory)

    # Covert temp files
    for filename in os.listdir(temp_directory):
        # Find csv then covert to numpy
        if '.csv' in filename:
            date_convert = lambda x: (eastern.localize(datetime.strptime(str(x, 'utf-8'), '%Y%m%d %H%M%S'))).timestamp()
            nparray = genfromtxt('./temp/'+filename, delimiter=';', converters={0: date_convert}, dtype=[('utc_timestamp', int), ('open', float), ('high', float), ('low', float), ('close', float)], usecols=(0, 1, 2, 3, 4))
            # print(nparray.shape)
            with open(output_file_path, 'wb') as f:
                np.save(f, nparray)
            break
    return output_file_path

def set_percentage_prevent():
    pass

def log_text_prevent():
    pass

pair_list = []

with open(os.path.dirname(__file__)+'/../historical_pair_data_fetcher/pairs.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader, None)
    for row in reader:
        currency_pair_name, pair, history_first_trading_date = row
        pair_list.append(pair)

def get_historical_pair_list():
    return pair_list

def update_historical_pair_data(set_percentage=set_percentage_prevent, log_text=log_text_prevent):
    set_percentage(0)
    log_text('Updating...\n')

    year_now = datetime.now().year
    month_now = datetime.now().month

    with open(os.path.dirname(__file__)+'/../historical_pair_data_fetcher/pairs.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        total_row_count = 0

        # count total
        with open(os.path.dirname(__file__)+'/../historical_pair_data_fetcher/pairs.csv', 'r') as f2:
            reader_counter = csv.reader(f2, delimiter=',')
            total_row_count = sum(1 for row in reader_counter)-1

        for row_index, row in enumerate(reader):
            set_percentage(0)
            currency_pair_name, pair, history_first_trading_date = row
            history_first_trading_year = int(history_first_trading_date[0:4])
            year = history_first_trading_year
            percentage_per_year = 100/(datetime.now().year - year + 1)
            log_text('Updating ' + currency_pair_name + '.\n')
            output_directory = os.path.join(pair_data_directory, pair)
            try:
                while True:
                    could_download_full_year = False
                    try:
                        log_text('Downloading pair: '+currency_pair_name+', year: '+str(year)+'. ')
                        update_a_historical_pair_data(year=str(year),
                                                      pair=pair,
                                                      output_directory=output_directory)
                        log_text('Downloaded.\n')
                        could_download_full_year = True
                    except AssertionError:
                        log_text('Downloading by month.\n')  # lets download it month by month.
                    month_count = 12
                    if year == year_now:
                        month_count = month_now
                    month = 1
                    while not could_download_full_year and month <= 12:
                        if month > month_count:
                            raise
                        log_text('Downloading pair: '+currency_pair_name+', year: '+str(year)+', month: '+str(month)+'. ')
                        try:
                            update_a_historical_pair_data(year=str(year),
                                                          month=str(month),
                                                          pair=pair,
                                                          output_directory=output_directory,
                                                          download_again=True)
                        except Exception as e:
                            log_text('\n')
                            log_text(str(e))
                            log_text('\nSkiped.\n')
                            raise
                        log_text('Downloaded.\n')

                        set_percentage(int(percentage_per_year*(year-(history_first_trading_year-1)-1) + (percentage_per_year/month_count)*month))
                        month += 1

                    set_percentage(int(percentage_per_year*(year-(history_first_trading_year-1))))
                    year += 1
            except Exception as e:
                # print(e)
                set_percentage(100)
                log_text('Done for currency '+  currency_pair_name+ '(' + str(row_index+1) +'/'+str(total_row_count)+').\n\n')

    log_text('Finished.\n')
    set_percentage(100)

def get_historical_pair_data(pair, from_datetime, to_datetime):
    if from_datetime > to_datetime:
        return None

    timestamp_lower = datetime.timestamp(from_datetime)
    timestamp_higher = datetime.timestamp(to_datetime)
    # print(timestamp_higher)
        # raise Exception('from_datetime > to_datetime!')
    pair_data_path = os.path.join(pair_data_directory, pair)
    pair_data_list = os.listdir(pair_data_path)

    filtered_array_list = []

    for year in range(from_datetime.year, to_datetime.year+1):
        # Check year data first

        substring = '_'+str(year)+'.npy'
        strings_with_substring = [string for string in pair_data_list if substring in string]
        # Full year
        if len(strings_with_substring):
            # print()
            filename = strings_with_substring[0]
            with open(os.path.join(pair_data_path, filename), 'rb') as f:
                year_array = np.load(f)
                filtered_array = year_array[year_array['utc_timestamp']>=timestamp_lower]
                filtered_array = filtered_array[filtered_array['utc_timestamp']<=timestamp_higher]
                # print(filtered_array)
                filtered_array_list.append(filtered_array)
        # Month
        else:
            month = 1
            while (year == to_datetime.year and month<=to_datetime.month) or (year < to_datetime.year and month<=12):
                substring = '_'+str(year)+str(month).zfill(2)+'.npy'
                strings_with_substring = [string for string in pair_data_list if substring in string]
                if len(strings_with_substring):
                    filename = strings_with_substring[0]
                    with open(os.path.join(pair_data_path, filename), 'rb') as f:
                        month_array = np.load(f)
                        filtered_array = month_array[month_array['utc_timestamp']>=timestamp_lower]
                        filtered_array = filtered_array[filtered_array['utc_timestamp']<=timestamp_higher]
                        # print(filtered_array)
                        filtered_array_list.append(filtered_array)
                    month += 1
                else:
                    break

    return np.concatenate(filtered_array_list)

def get_historical_pair_data_pandas(pair, from_datetime, to_datetime, target_timezone='UTC'):
    df =  pd.DataFrame(get_historical_pair_data(pair, from_datetime, to_datetime))
    df['utc_timestamp']= pd.DatetimeIndex(pd.to_datetime(df['utc_timestamp'], unit='s')).tz_localize('UTC').tz_convert(target_timezone)
    df_new = df.rename(columns={'utc_timestamp': 'timestamp'})
    # print(df_new['timestamp'])
    return df_new

def plot_historical_pair_data(pair, from_datetime, to_datetime, target_timezone='UTC'):

    quotes = get_historical_pair_data_pandas(pair, from_datetime, to_datetime, target_timezone)
    fig = go.Figure(data=[go.Candlestick(x=quotes['timestamp'],
            open=quotes['open'],
            high=quotes['high'],
            low=quotes['low'],
            close=quotes['close'])])
    fig.update_layout(
        title='Historical chart of "'+pair+'". In "'+target_timezone+'" timezone.',
        yaxis_title=pair
    )
    # fig.update_yaxes(autorange=True)
    fig.show()
