#!/usr/bin/python3

import json, os, sys

sys.path.append('./trading_agents')

from datetime import datetime
from coin_wizard.historical_pair_data import update_historical_pair_data
from coin_wizard.main.event_manager import EventManager
from prompt_toolkit.shortcuts import radiolist_dialog, progress_dialog

print('')
print('CoinWizard by noowyee')
print('-')

cwd = os.getcwd()
print('Current working directory:', cwd)

settings = None

with open('settings.json') as settings_file:
    settings = json.load(settings_file)

def test(num):
    pass

def save_settings():
    with open('settings.json', 'w') as outfile:
        json.dump(settings, outfile)

trading_agent_mode = "STOP"

def stop_agent():
    global trading_agent_mode
    if trading_agent_mode == "RUN":
        trading_agent.stop_running()
    elif trading_agent_mode == "TRAIN":
        trading_agent.stop_training()
    elif trading_agent_mode == "TEST":
        trading_agent.stop_testing()
    trading_agent_mode = "STOP"

def create_agent(trading_agent_name):
    trading_agent_module = __import__(trading_agent_name)
    print('Selected trading agent:', trading_agent_module)
    return trading_agent_module.TradingAgent({
    'test': test})


trading_agent = create_agent(settings['trading_agent'])

def start():
    global trading_agent
    while True:
        selections = [
            (0, 'Run    trading agent'),
            (1, 'Train  trading agent'),
            (2, 'Test   trading agent by backtesting with historical pair data.'),
            (3, 'Change trading agent.'),
            (10, 'Plot   historical pair data'),
            (11, 'Update historical pair data'),
            (99, 'Leave'),
        ]
        answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'"). \n', values = selections).run()

        if answer == 99:
            print('Good bye!')
            break

        elif answer == 0:
            trading_agent_mode = "RUN"
            trading_agent.run("Broker API")
            stop_agent()

        elif answer == 1:
            trading_agent_mode = "TRAIN"
            trading_agent.train()
            stop_agent()

        elif answer == 2:
            trading_agent_mode = "TEST"
            trading_agent.test("Backtest Broker API")
            stop_agent()

        elif answer == 3:
            agent_selections = [(filename, filename) for filename in os.listdir('./trading_agents')]
            agent_answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'"). \n', values = agent_selections).run()
            settings['trading_agent'] = agent_answer
            stop_agent()
            trading_agent = create_agent(agent_answer)
            save_settings()

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
