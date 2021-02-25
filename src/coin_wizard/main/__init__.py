#!/usr/bin/python3

import json
import os
import sys
import csv

sys.path.append('./trading_agents')

from datetime import datetime
from coin_wizard.historical_pair_data_fetcher import download_hist_data
from coin_wizard.historical_pair_data_fetcher.api import Platform
from coin_wizard.main.event_manager import EventManager
from prompt_toolkit.shortcuts import radiolist_dialog, progress_dialog

print('')
print('CoinWizard by noowyee')
print('-')

cwd = os.getcwd()
print('Current working directory:', cwd)

def update_historical_pair_data(set_percentage, log_text):
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
            output_folder = 'pair_data/'+pair
            try:
                while True:
                    could_download_full_year = False
                    try:
                        log_text('Downloading pair: '+currency_pair_name+', year: '+str(year)+'. ')
                        download_hist_data(year=str(year),
                                                      pair=pair,
                                                      output_directory=output_folder,
                                                      verbose=False)
                        log_text('Downloaded.\n')
                        could_download_full_year = True
                    except AssertionError:
                        log_text('Downloading by month.\n')  # lets download it month by month.
                    month = 1
                    while not could_download_full_year and month <= 12:
                        if month > month_now and year == year_now:
                            raise
                        log_text('Downloading pair: '+currency_pair_name+', year: '+str(year)+', month: '+str(month)+'. ')
                        try:
                            download_hist_data(year=str(year),
                                                          month=str(month),
                                                          pair=pair,
                                                          output_directory=output_folder,
                                                          download_again=True,
                                                          verbose=False)
                        except Exception:
                            log_text('Skiped.\n')
                            raise
                        log_text('Downloaded.\n')
                        set_percentage(int(percentage_per_year*(year-(history_first_trading_year-1)-1) + (percentage_per_year/month_now)*month))
                        month += 1

                    set_percentage(int(percentage_per_year*(year-(history_first_trading_year-1))))
                    year += 1
            except Exception:
                set_percentage(100)
                log_text('Done for currency '+  currency_pair_name+ '(' + str(row_index+1) +'/'+str(total_row_count)+').\n\n')

    log_text('Finished.\n')
    set_percentage(100)

def test(num):
    pass

def start():
    # Def
    settings = None
    trading_agent = None
    trading_agent_thread = None
    trading_agent_event_manager = EventManager()

    with open('settings.json') as settings_file:
        settings = json.load(settings_file)
        trading_agent = settings['trading_agent']
        trading_agent_module = __import__(trading_agent)
        print('Selected trading agent:', trading_agent_module)
        trading_agent = trading_agent_module.TradingAgent({
        'test': test, 'on': trading_agent_event_manager.on})

    while True:
        questions = [
            (0, 'Run    trading agent'),
            (1, 'Train  trading agent'),
            (2, 'Test   trading agent by simulation with historical pair data.'),
            (3, 'Change trading agent.'),
            (10, 'Plot   historical pair data'),
            (11, 'Update historical pair data'),
            (99, 'Leave'),
        ]
        answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'")', values = questions).run()

        if answer == 99:
            print('Good bye!')
            break
        elif answer == 11:
            progress_dialog(
                title="Updating historical data",
                text= datetime.now().strftime("Today's date in your timezone: %Y %b %d."),
                run_callback=update_historical_pair_data,
            ).run()
            # break

        elif answer == 10:
            pass
        else:
            print('Good bye!')
            break
