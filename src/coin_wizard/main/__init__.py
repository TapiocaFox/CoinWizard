#!/usr/bin/python3

import json
import os
import sys
sys.path.append('./trading_agents')

from coin_wizard.historical_pair_data_fetcher import download_hist_data
from coin_wizard.historical_pair_data_fetcher.api import Platform
from coin_wizard.main.event_manager import EventManager
from PyInquirer import prompt, Separator

print('')
print('CoinWizard by noowyee')
print('-')

cwd = os.getcwd()
print('Current working directory:', cwd)

def test(num):
    print('test', num)

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
            {
                'type': 'list',
                'name': 'operation',
                'message': 'What do you want to do?',
                'choices': [
                    'Run trading agent',
                    'Train trading agent',
                    'Test trading agent',
                    'Plot historical data',
                    'Update historical data',
                    'Leave'
                ],
            }
        ]
        answers = prompt(questions)
        if not len(answers):
            print('Mouse event not supported yet by PyInquirer. (Issue #41)')
            pass
        elif answers['operation'] == 'Leave':
            print('Good bye!')
            break
        elif answers['operation'] == 'Update historical data':
            print('-', download_hist_data(
                  year='2018',
                  output_directory='pair_data',
                  verbose=True,
                  # platform=Platform.META_STOCK
                  ))
        elif answers['operation'] == 'Plot historical data':
            pass
