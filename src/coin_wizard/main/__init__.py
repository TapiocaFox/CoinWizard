#!/usr/bin/python3

import json, os, sys, pytz

sys.path.append('./trading_agents')

from datetime import datetime

from coin_wizard.historical_pair_data import update_historical_pair_data
from coin_wizard.main.event_manager import EventManager
from coin_wizard.historical_pair_data import plot_historical_pair_data

from prompt_toolkit.shortcuts import radiolist_dialog, progress_dialog, input_dialog
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory

eastern = pytz.timezone('US/Eastern')
prompt_history = InMemoryHistory()

for tz in pytz.all_timezones:
    prompt_history.append_string(tz)

session = PromptSession(
    history=prompt_history,
    auto_suggest=AutoSuggestFromHistory(),
    enable_history_search=True,
)

print('')
print('CoinWizard by noowyee')
print('-')

cwd = os.getcwd()
print('Current working directory:', cwd)

states = {
    "latest_historical_pair_data_update": "no record"
}
settings = None

if not os.path.exists('trading_agents_files'):
    os.makedirs('trading_agents_files')

with open('settings.json') as settings_file:
    settings = json.load(settings_file)

def save_settings():
    with open('settings.json', 'w') as outfile:
        json.dump(settings, outfile)

def save_states():
    with open('states.json', 'w') as outfile:
        json.dump(states, outfile)

if not os.path.exists('states.json'):
    save_states()
else:
    with open('states.json') as states_file:
        states = json.load(states_file)

def test(num):
    pass


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
    if not os.path.exists(os.path.join(cwd, 'trading_agents_files', trading_agent_name)):
        os.makedirs(os.path.join(cwd, 'trading_agents_files', trading_agent_name))
    return trading_agent_module.TradingAgent(os.path.join(cwd, 'trading_agents_files', trading_agent_name))


trading_agent = create_agent(settings['trading_agent'])

def start():
    global trading_agent
    while True:
        selections = [
            (0, 'Run    trading agent.'),
            (1, 'Train  trading agent.'),
            (2, 'Test   trading agent by backtesting with historical pair data.'),
            (3, 'Change trading agent.'),
            (10, 'Plot   historical pair data.'),
            (11, 'Update historical pair data. (Latest: '+states['latest_historical_pair_data_update']+')'),
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

        elif answer == 10:
            print('\n=== Plot historical pair data ===\n')
            pair = session.prompt("Pair: ", default="eurusd")
            timezone = session.prompt("Output Timezone: ", default="US/Eastern")
            print('\nFrom date (Timezone/Year/Month/Day):')
            from_timezone = session.prompt("Timezone: ", default="US/Eastern")
            from_year = int(session.prompt("Year: ", default="2016"))
            from_month = int(session.prompt("Month: ", default="1"))
            from_day = int(session.prompt("Day: ", default="1"))
            print('\nTo date (Timezone/Year/Month/Day):')
            to_timezone = session.prompt("Timezone: ", default="US/Eastern")
            to_year = int(session.prompt("Year: ", default="2016"))
            to_month = int(session.prompt("Month: ", default="1"))
            to_day = int(session.prompt("Day: ", default="2"))
            plot_historical_pair_data(pair, eastern.localize(datetime(from_year, from_month, from_day, 0, 0)), eastern.localize(datetime(to_year, to_month, to_day, 23, 59)), timezone)

        elif answer == 11:
            progress_dialog(
                title="Updating historical data",
                text= datetime.now().strftime("Today's date in your timezone: %Y %m %d."),
                run_callback=update_historical_pair_data,
            ).run()
            states['latest_historical_pair_data_update'] = datetime.now().strftime("%Y,%m,%d, %H:%M:%S")
            save_states()

        else:
            print('Good bye!')
            break
