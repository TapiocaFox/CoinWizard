#!/usr/bin/python3

import json, os, sys, pytz

sys.path.append('./trading_agents')
sys.path.append('./coin_wizard/broker_platforms')

from datetime import datetime

from coin_wizard.historical_pair_data import update_historical_pair_data, plot_historical_pair_data, get_historical_pair_list
from coin_wizard.main.event_manager import EventManager

from prompt_toolkit.shortcuts import radiolist_dialog, progress_dialog, input_dialog
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory

prompt_history = InMemoryHistory()

for tz in pytz.all_timezones:
    prompt_history.append_string(tz)

for pair in get_historical_pair_list():
    prompt_history.append_string(pair)

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
    "latest_historical_pair_data_update": "no record",
    "latest_plot_settings": {
        "pair": "eurusd",
        "timezone": "US/Eastern",

        "from_timezone": "US/Eastern",
        "from_year": "2016",
        "from_month": "6",
        "from_day": "1",

        "to_timezone": "US/Eastern",
        "to_year": "2016",
        "to_month": "6",
        "to_day": "1"
    },
    "broker_platform_settings_dict": {

    }
}

settings = None

if not os.path.exists('trading_agents_files'):
    os.makedirs('trading_agents_files')

with open('settings.json') as settings_file:
    settings = json.load(settings_file)

def save_settings():
    with open('settings.json', 'w') as outfile:
        json.dump(settings, outfile, indent=2)

def save_states():
    with open('states.json', 'w') as outfile:
        json.dump(states, outfile, indent=2)

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

# def create_broker_platform(broker_platform_name):
#     broker_platform_module = __import__(broker_platform_name)
#     print('Selected trading agent:', trading_agent_module)
#     # if not os.path.exists(os.path.join(cwd, 'broker_platforms_files', broker_platform_name)):
#     #     os.makedirs(os.path.join(cwd, 'broker_platforms_files', broker_platform_name))
#     return broker_platform_module.TradingAgent()

trading_agent = create_agent(settings['trading_agent'])
broker_platform_module = __import__(settings['broker_platform']).BrokerEventLoopAPI

# print(broker_platform_module.broker_settings_fields)

def before_broker_platform_loop():
    pass

def after_broker_platform_loop():
    pass

def start():
    global trading_agent
    global broker_platform_module
    while True:
        selections = [
            (0, 'Run    trading agent.'),
            (1, '[x] Train  trading agent.'),
            (2, '[x] Test   trading agent by backtesting with historical pair data.'),
            (3, 'Change  trading agent.'),
            (4, 'Change broker platform.'),
            (5, 'Set    broker platform settings.'),
            (6, '[x] Plot   broker platform realtime pair data.'),
            (10, 'Plot   historical pair data.'),
            (11, 'Plot   latest historical pair data.'),
            (12, 'Update historical pair data. (Latest: '+states['latest_historical_pair_data_update']+')'),
            (13, '[x] Select which historical pair data to be followed.'),
            (99, 'Leave'),
        ]
        answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'"). \nBroker platform(Current: "'+ settings['broker_platform'] +'")', values = selections).run()

        if answer == 99:
            print('Good bye!')
            break

        elif answer == 0:
            trading_agent_mode = "RUN"
            broker_platform_settings = states["broker_platform_settings_dict"][settings['broker_platform']]
            trading_agent.run(broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings))
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
            agent_answer = radiolist_dialog(title='Trading agent', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'"). \n', values = agent_selections).run()
            if agent_answer == None:
                break
            settings['trading_agent'] = agent_answer
            stop_agent()
            trading_agent = create_agent(agent_answer)
            save_settings()

        elif answer == 4:
            bp_selections = [(filename, filename) for filename in os.listdir('./coin_wizard/broker_platforms')]
            bp_answer = radiolist_dialog(title='Broker platform', text='What do you want to do? \Broker platform(Current: "'+ settings['broker_platform'] +'"). \n', values = bp_selections).run()
            if bp_answer == None:
                break
            settings['broker_platform'] = bp_answer
            broker_platform_module = __import__(bp_answer).BrokerEventLoopAPI
            save_settings()

        elif answer == 5:
            bp = settings['broker_platform']
            bp_settings = {}
            print('\n=== "Broker platforms(Current: "'+bp+'")" settings ===')
            for field in broker_platform_module.broker_settings_fields:
                bp_settings[field] = session.prompt("    "+field + ": ")
            states["broker_platform_settings_dict"][bp] = bp_settings
            save_states()

        elif answer == 10:
            print('\n=== "Plot historical pair data" settings ===')
            states["latest_plot_settings"]["pair"] = pair = session.prompt("  Pair: ", default=str(states["latest_plot_settings"]["pair"]))
            states["latest_plot_settings"]["timezone"] = timezone = session.prompt("  Output Timezone: ", default=str(states["latest_plot_settings"]["timezone"]))
            print('\n  From date (Timezone/Year/Month/Day):')
            states["latest_plot_settings"]["from_timezone"] = session.prompt("    Timezone: ", default=str(states["latest_plot_settings"]["from_timezone"]))
            from_timezone = pytz.timezone(states["latest_plot_settings"]["from_timezone"])
            states["latest_plot_settings"]["from_year"] = from_year = int(session.prompt("    Year: ", default=str(states["latest_plot_settings"]["from_year"])))
            states["latest_plot_settings"]["from_month"] = from_month = int(session.prompt("    Month: ", default=str(states["latest_plot_settings"]["from_month"])))
            states["latest_plot_settings"]["from_day"] = from_day = int(session.prompt("    Day: ", default=str(states["latest_plot_settings"]["from_day"])))
            print('\n  To date (Timezone/Year/Month/Day):')
            states["latest_plot_settings"]["to_timezone"] = session.prompt("    Timezone: ", default=str(states["latest_plot_settings"]["to_timezone"]))
            to_timezone = pytz.timezone(states["latest_plot_settings"]["to_timezone"])
            states["latest_plot_settings"]["to_year"] = to_year = int(session.prompt("    Year: ", default=str(states["latest_plot_settings"]["to_year"])))
            states["latest_plot_settings"]["to_month"] = to_month = int(session.prompt("    Month: ", default=str(states["latest_plot_settings"]["to_month"])))
            states["latest_plot_settings"]["to_day"] = to_day = int(session.prompt("    Day: ", default=str(states["latest_plot_settings"]["to_day"])))
            save_states()
            print('')
            print('Ploting...')
            plot_historical_pair_data(pair, from_timezone.localize(datetime(from_year, from_month, from_day, 0, 0)), to_timezone.localize(datetime(to_year, to_month, to_day, 23, 59)), timezone)
            print('Ploted.')

        elif answer == 11:
            pair = states["latest_plot_settings"]["pair"]
            timezone = states["latest_plot_settings"]["timezone"]

            from_timezone = pytz.timezone(states["latest_plot_settings"]["from_timezone"])
            from_year = int(states["latest_plot_settings"]["from_year"])
            from_month = int(states["latest_plot_settings"]["from_month"])
            from_day = int(states["latest_plot_settings"]["from_day"])

            to_timezone = pytz.timezone(states["latest_plot_settings"]["to_timezone"])
            to_year = int(states["latest_plot_settings"]["to_year"])
            to_month = int(states["latest_plot_settings"]["to_month"])
            to_day = int(states["latest_plot_settings"]["to_day"])

            print('')
            print('Ploting...')
            plot_historical_pair_data(pair, from_timezone.localize(datetime(from_year, from_month, from_day, 0, 0)), to_timezone.localize(datetime(to_year, to_month, to_day, 23, 59)), timezone)
            print('Ploted.')

        elif answer == 12:
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
