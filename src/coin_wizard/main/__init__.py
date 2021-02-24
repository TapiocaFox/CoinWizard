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
    with open(os.path.dirname(__file__)+'/../historical_pair_data_fetcher/pairs.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            set_percentage(0)
            currency_pair_name, pair, history_first_trading_month = row
            year = int(history_first_trading_month[0:4])
            percentage_per_year = 100/(datetime.now().year - year)
            log_text('Updating ' + currency_pair_name + '.\n')
            output_folder = 'pair_data/'+pair
            try:
                while True:
                    could_download_full_year = False
                    try:
                        log_text('Downloaded ' + download_hist_data(year=str(year),
                                                      pair=pair,
                                                      output_directory=output_folder,
                                                      verbose=False)+ '.\n')
                        could_download_full_year = True
                    except AssertionError:
                        pass  # lets download it month by month.
                    month = 1
                    while not could_download_full_year and month <= 12:
                        log_text('Downloaded ' + download_hist_data(year=str(year),
                                                      month=str(month),
                                                      pair=pair,
                                                      output_directory=output_folder,
                                                      verbose=False)+ '.\n')
                        month += 1
                    year += 1
                    set_percentage(int(percentage_per_year*(year-int(history_first_trading_month[0:4]))))
            except Exception:
                set_percentage(100)
                log_text('Done for currency '+  currency_pair_name+ '.\n')

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
        answer = radiolist_dialog(title='What do you want to do?', text='Trading agent(Current: "'+ settings['trading_agent'] +'")', values = questions).run()

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
