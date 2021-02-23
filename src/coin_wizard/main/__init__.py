#!/usr/bin/python3

import json
import os
import sys
sys.path.append('./trading_agents')

from coin_wizard.pair_data_fetcher import PairDataFetcher
from coin_wizard.main.event_manager import EventManager

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
    trading_agent_module= None
    trading_agent_thread = None
    pair_data_fetcher_thread = PairDataFetcher()
    event_manager = EventManager()

    with open('settings.json') as settings_file:
        settings = json.load(settings_file)
        trading_agent = settings['trading_agent']
        trading_agent_module = __import__(trading_agent)
        print('Selected trading agent:', trading_agent_module)
        trading_agent_thread = trading_agent_module.TradingAgent({
        'test': test, 'on': event_manager.on})

    pair_data_fetcher_thread.start()
    trading_agent_thread.start()
    event_manager.emit('CliMessage', [123])
    trading_agent_thread.join()
