#!/usr/bin/python3

import json, os, sys, pytz, signal
import coin_wizard.plotter as plotter

sys.path.append('./trading_agents')
sys.path.append('./coin_wizard/broker_platforms')
sys.path.append('./coin_wizard/notification_service_providers')

from datetime import datetime, timedelta

from coin_wizard.historical_pair_data import update_historical_pair_data, plot_historical_pair_data, get_historical_pair_list
from coin_wizard.main.event_manager import EventManager
from coin_wizard.utils import translate_pair_to_splited, translate_pair_to_unsplited

from prompt_toolkit.shortcuts import radiolist_dialog, progress_dialog, input_dialog
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory

time_delta_1_days = timedelta(days=1)

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
    "latest_broker_plot_settings": {
        "pair": "EUR_USD",
        "timezone": "US/Eastern",
        "counts": 1000
    },
    "broker_platform_settings_dict": {

    }
    ,
    "notification_service_provider_settings_dict": {

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
        loaded_states = json.load(states_file)
        for key in loaded_states:
            states[key] = loaded_states[key]
        save_states()

def test(num):
    pass

trading_agent_mode = "STOP"
broker_platform_module = __import__(settings['broker_platform']).BrokerEventLoopAPI
test_train_broker_platform_module = __import__(settings['test_train_broker_platform']).BrokerEventLoopAPI
broker_platform = None
notification_service_provider_module = __import__(settings['notification_service_provider']).NotificationServiceProvider
if settings['notification_service_provider'] not in states["notification_service_provider_settings_dict"]:
    states["notification_service_provider_settings_dict"][settings['notification_service_provider']] = {}
save_states()
nsp = None

def stop_agent():
    global trading_agent_mode
    # print(trading_agent_mode)

    if trading_agent_mode == "RUN":
        trading_agent.stop_running(broker_platform)
        broker_platform._loop()
        push_trade_agent_stop_notification()

    elif trading_agent_mode == "TRAIN":
        trading_agent.stop_training(broker_platform)
        broker_platform._loop()
        push_trade_agent_stop_notification()

    elif trading_agent_mode == "TEST":
        trading_agent.stop_testing(broker_platform)
        broker_platform._loop()
        push_trade_agent_stop_notification()

    trading_agent_mode = "STOP"

def create_agent(trading_agent_name):
    trading_agent_module = __import__(trading_agent_name)
    print('Selected trading agent:', trading_agent_module)
    if not os.path.exists(os.path.join(cwd, 'trading_agents_files', trading_agent_name)):
        os.makedirs(os.path.join(cwd, 'trading_agents_files', trading_agent_name))
    return trading_agent_module.TradingAgent(os.path.join(cwd, 'trading_agents_files', trading_agent_name))

trading_agent = None
# def create_broker_platform(broker_platform_name):
#     broker_platform_module = __import__(broker_platform_name)
#     print('Selected trading agent:', trading_agent_module)
#     # if not os.path.exists(os.path.join(cwd, 'broker_platforms_files', broker_platform_name)):
#     #     os.makedirs(os.path.join(cwd, 'broker_platforms_files', broker_platform_name))
#     return broker_platform_module.TradingAgent()

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    if broker_platform!= None:
        broker_platform._stop()

signal.signal(signal.SIGINT, signal_handler)
# print(broker_platform_module.broker_settings_fields)

def before_broker_platform_loop():
    # print(1234)
    pass

def after_broker_platform_loop():
    pass

current_broker_platform_name = settings['broker_platform']
# current_trading_agent_name = settings['trading_agent']

def order_canceled_listener(order, reason):
    global nsp
    nsp.addLine('reason: %s' % (reason))
    nsp.addLine('order settings: %s' % (json.dumps(order.order_settings, indent=2)))
    nsp.addLine('trade settings: %s' % (json.dumps(order.trade_settings, indent=2)))
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Order canceled (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def order_filled_listener(order_be_filled, trade_with_order):
    global nsp
    nsp.addLine('instrument: %s' % (order_be_filled.instrument_name))
    nsp.addLine('order settings: %s' % (json.dumps(order_be_filled.order_settings, indent=2)))
    nsp.addLine('trade settings: %s' % (json.dumps(order_be_filled.trade_settings, indent=2)))
    if trade_with_order != None:
        nsp.addLine('open price: %s' % (trade_with_order.open_price))
        nsp.addLine('real trade settings: %s' % (json.dumps(trade_with_order.trade_settings, indent=2)))
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Order filled (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def trade_closed_listener(trade, realized_pl, close_price, spread, timestamp):
    global nsp
    global broker_platform
    nsp.addLine('instrument: %s' % trade.instrument_name)
    nsp.addLine('realized pl: %.5f' % (realized_pl))
    nsp.addLine('open price: %s' % (trade.open_price))
    nsp.addLine('close price: %.5f' % (close_price))
    nsp.addLine('trade settings: %s' % (json.dumps(trade.trade_settings, indent=2)))
    nsp.addLine('account balance: %.5f' % broker_platform.account.getBalance())
    nsp.addLine('account unrealized pl: %.5f' % broker_platform.account.getUnrealizedPL())
    nsp.addLine('account margin available: %.5f' % broker_platform.account.getMarginAvailable())
    nsp.addLine('account margin used: %.5f' % broker_platform.account.getMarginUsed())
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Trade closed (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def trade_reduced_listener(trade, units, realized_pl, close_price, spread, timestamp):
    global nsp
    global broker_platform
    nsp.addLine('instrument: %s' % trade.instrument_name)
    nsp.addLine('units: %.5f' % units)
    nsp.addLine('realized pl: %.5f' % (realized_pl))
    nsp.addLine('open price: %s' % (trade.open_price))
    nsp.addLine('close price: %.5f' % (close_price))
    nsp.addLine('trade settings: %s' % (json.dumps(trade.trade_settings, indent=2)))
    nsp.addLine('account balance: %.5f' % broker_platform.account.getBalance())
    nsp.addLine('account unrealized pl: %.5f' % broker_platform.account.getUnrealizedPL())
    nsp.addLine('account margin available: %.5f' % broker_platform.account.getMarginAvailable())
    nsp.addLine('account margin used: %.5f' % broker_platform.account.getMarginUsed())
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Trade reduced (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def push_trade_agent_start_notification():
    nsp.addLine('account balance: %.5f' % (broker_platform.account.balance))
    nsp.addLine('account currency: %s' % (broker_platform.account.currency))
    nsp.addLine('account margin rate: %.5f' % (broker_platform.account.margin_rate))
    nsp.addLine('account margin used: %.5f' % (broker_platform.account.margin_used))
    nsp.addLine('account margin available: %.5f' % (broker_platform.account.margin_available))
    nsp.addLine('account unrealized pl: %.5f' % (broker_platform.account.unrealized_pl))
    nsp.addLine('account trades counts: %3d' % (len(broker_platform.account.trades)))
    nsp.addLine('account orders count: %3d' % (len(broker_platform.account.orders)))
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Trade agent run (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def push_trade_agent_train_test_notification():
    nsp.addLine('account balance: %.5f' % (broker_platform.account.balance))
    nsp.addLine('account currency: %s' % (broker_platform.account.currency))
    nsp.addLine('account margin rate: %.5f' % (broker_platform.account.margin_rate))
    nsp.addLine('account margin used: %.5f' % (broker_platform.account.margin_used))
    nsp.addLine('account margin available: %.5f' % (broker_platform.account.margin_available))
    nsp.addLine('account unrealized pl: %.5f' % (broker_platform.account.unrealized_pl))
    nsp.addLine('account trades counts: %3d' % (len(broker_platform.account.trades)))
    nsp.addLine('account orders count: %3d' % (len(broker_platform.account.orders)))
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Trade agent train/test (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def push_trade_agent_stop_notification():
    nsp.addLine('account balance: %.5f' % (broker_platform.account.balance))
    nsp.addLine('account currency: %s' % (broker_platform.account.currency))
    nsp.addLine('account margin rate: %.5f' % (broker_platform.account.margin_rate))
    nsp.addLine('account margin used: %.5f' % (broker_platform.account.margin_used))
    nsp.addLine('account margin available: %.5f' % (broker_platform.account.margin_available))
    nsp.addLine('account unrealized pl: %.5f' % (broker_platform.account.unrealized_pl))
    nsp.addLine('account trades counts: %3d' % (len(broker_platform.account.trades)))
    nsp.addLine('account orders count: %3d' % (len(broker_platform.account.orders)))
    # nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.addLine('push date: %s' % (datetime.now().isoformat()))
    nsp.push('Trade agent stop (%s/%s)'%(current_broker_platform_name, settings['trading_agent']))

def start():
    global trading_agent_mode
    global trading_agent
    global broker_platform_module
    global test_train_broker_platform_module
    global notification_service_provider_module
    global broker_platform
    global current_broker_platform_name
    global nsp

    while True:
        selections = [
            (0, 'Run    trading agent. (Broker platform: '+settings['broker_platform']+')'),
            (1, 'Train  trading agent. (Broker platform: '+settings['test_train_broker_platform']+')'),
            (2, 'Test   trading agent. (Broker platform: '+settings['test_train_broker_platform']+')'),
            (3, 'Change  trading agent. ('+settings['trading_agent']+')'),
            (4, 'Change broker platform. ('+settings['broker_platform']+')'),
            (5, 'Change test/train broker platform. ('+settings['test_train_broker_platform']+')'),
            (6, 'Set    broker platform settings. ('+settings['broker_platform']+')'),
            (7, 'Set    test/train broker platform settings. ('+settings['test_train_broker_platform']+')'),
            (8, 'Plot   broker platform recent 1M pair data.'),
            (9, 'Plot   broker platform recent 1M pair data with previous settings.'),
            (10, 'Plot   historical 1M pair data.'),
            (11, 'Plot   previous historical 1M pair data.'),
            (12, 'Update historical pair data. (Latest: '+states['latest_historical_pair_data_update']+')'),
            (20, 'Change notification service provider. ('+settings['notification_service_provider']+')'),
            (21, 'Set    notification service provider settings. ('+settings['notification_service_provider']+')'),
            # (13, '[x] Select which historical pair data to be followed.'),
            (99, 'Leave'),
        ]
        # answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent']
        # +'"). \nBroker platform(Current: "'+ settings['broker_platform'] +'"). \nTest train broker platform(Current: "'+ settings['test_train_broker_platform']+'").', values = selections).run()
        answer = radiolist_dialog(title='CoinWizard by noowyee', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent']
        +'").', values = selections).run()
        if answer == 99:
            print('Good bye!')
            break

        elif answer == 0:
            trading_agent_mode = "RUN"
            trading_agent = create_agent(settings['trading_agent'])
            current_broker_platform_name = settings['broker_platform']
            broker_platform_settings = states["broker_platform_settings_dict"][settings['broker_platform']]
            print('Initializing broker platform('+settings['broker_platform']+')...')
            nsp_settings = states["notification_service_provider_settings_dict"][settings['notification_service_provider']]
            nsp = notification_service_provider_module(nsp_settings)
            broker_platform = broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings, nsp)
            broker_platform.order_canceled_listener = order_canceled_listener
            broker_platform.order_filled_listener = order_filled_listener
            broker_platform.trade_closed_listener = trade_closed_listener
            broker_platform.trade_reduced_listener = trade_reduced_listener
            push_trade_agent_start_notification()
            print('Running trading agent...')
            trading_agent.run(broker_platform)
            print('Starting broker platform event loop...')
            print('Press Ctrl+C to end loop.')
            broker_platform._run_loop()
            stop_agent()

        elif answer == 1:
            trading_agent_mode = "TRAIN"
            trading_agent = create_agent(settings['trading_agent'])
            current_broker_platform_name = settings['test_train_broker_platform']
            broker_platform_settings = states["broker_platform_settings_dict"][settings['test_train_broker_platform']]
            print('Initializing test/train broker platform('+settings['test_train_broker_platform']+')...')
            nsp_settings = states["notification_service_provider_settings_dict"][settings['notification_service_provider']]
            nsp = notification_service_provider_module(nsp_settings)
            broker_platform = test_train_broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings, nsp)
            broker_platform.order_canceled_listener = order_canceled_listener
            broker_platform.order_filled_listener = order_filled_listener
            broker_platform.trade_closed_listener = trade_closed_listener
            broker_platform.trade_reduced_listener = trade_reduced_listener
            push_trade_agent_train_test_notification()
            print('Testing trading agent...')
            trading_agent.train(broker_platform)
            print('Starting test/train broker platform event loop...')
            print('Press Ctrl+C to end loop.')
            broker_platform._run_loop()
            stop_agent()

        elif answer == 2:
            trading_agent_mode = "TEST"
            trading_agent = create_agent(settings['trading_agent'])
            current_broker_platform_name = settings['test_train_broker_platform']
            broker_platform_settings = states["broker_platform_settings_dict"][settings['test_train_broker_platform']]
            print('Initializing test/train broker platform('+settings['test_train_broker_platform']+')...')
            nsp_settings = states["notification_service_provider_settings_dict"][settings['notification_service_provider']]
            nsp = notification_service_provider_module(nsp_settings)
            broker_platform = test_train_broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings, nsp)
            broker_platform.order_canceled_listener = order_canceled_listener
            broker_platform.order_filled_listener = order_filled_listener
            broker_platform.trade_closed_listener = trade_closed_listener
            broker_platform.trade_reduced_listener = trade_reduced_listener
            push_trade_agent_train_test_notification()
            print('Testing trading agent...')
            trading_agent.test(broker_platform)
            print('Starting test/train broker platform event loop...')
            print('Press Ctrl+C to end loop.')
            broker_platform._run_loop()
            stop_agent()

        elif answer == 3:
            agent_selections = [(filename, filename) for filename in os.listdir('./trading_agents')]
            agent_answer = radiolist_dialog(title='Trading agent', text='What do you want to do? \nTrading agent(Current: "'+ settings['trading_agent'] +'"). \n', values = agent_selections).run()
            if agent_answer == None:
                continue
            settings['trading_agent'] = agent_answer
            stop_agent()
            save_settings()

        elif answer == 4:
            bp_selections = [(filename, filename) for filename in os.listdir('./coin_wizard/broker_platforms')]
            bp_answer = radiolist_dialog(title='Broker platform', text='What do you want to do? \nBroker platform(Current: "'+ settings['broker_platform'] +'"). \n', values = bp_selections).run()
            if bp_answer == None:
                continue
            settings['broker_platform'] = bp_answer
            broker_platform_module = __import__(bp_answer).BrokerEventLoopAPI
            save_settings()

        elif answer == 5:
            bp_selections = [(filename, filename) for filename in os.listdir('./coin_wizard/broker_platforms')]
            bp_answer = radiolist_dialog(title='Test/train broker platform', text='What do you want to do? \nTest/train broker platform(Current: "'+ settings['test_train_broker_platform'] +'"). \n', values = bp_selections).run()
            if bp_answer == None:
                continue
            settings['test_train_broker_platform'] = bp_answer
            test_train_broker_platform_module = __import__(bp_answer).BrokerEventLoopAPI
            save_settings()

        elif answer == 6:
            bp = settings['broker_platform']
            bp_settings = {}
            print('\n=== "Broker platform(Current: "'+bp+'")" settings ===')
            for field in broker_platform_module.broker_settings_fields:
                bp_settings[field] = session.prompt("    "+field + ": ")
            states["broker_platform_settings_dict"][bp] = bp_settings
            save_states()

        elif answer == 7:
            bp = settings['test_train_broker_platform']
            bp_settings = {}
            print('\n=== "Test/train broker platform(Current: "'+bp+'")" settings ===')
            for field in test_train_broker_platform_module.broker_settings_fields:
                bp_settings[field] = session.prompt("    "+field + ": ")
            states["broker_platform_settings_dict"][bp] = bp_settings
            save_states()

        elif answer == 8:
            broker_platform_settings = states["broker_platform_settings_dict"][settings['broker_platform']]
            print('Initializing broker platform('+settings['broker_platform']+')...')
            broker_platform = broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings)
            print('\n=== "Plot broker platform recent pair data" settings ===')
            states["latest_broker_plot_settings"]["pair"] = pair = session.prompt("  Pair: ", default=str(states["latest_broker_plot_settings"]["pair"]))
            states["latest_broker_plot_settings"]["timezone"] = timezone = session.prompt("  Output Timezone: ", default=str(states["latest_broker_plot_settings"]["timezone"]))
            states["latest_broker_plot_settings"]["counts"] = counts = int(session.prompt("  Counts: ", default=str(states["latest_broker_plot_settings"]["counts"])))
            if pair.islower():
                pair = translate_pair_to_splited(pair)
            instrument = broker_platform.getInstrument(pair)
            save_states()
            print('')
            print('Ploting...')
            plotter.plot_candles(pair+' in '+timezone, instrument.getRecent1MCandles(counts), timezone)
            print('Ploted.')

        elif answer == 9:
            broker_platform_settings = states["broker_platform_settings_dict"][settings['broker_platform']]
            print('Initializing broker platform('+settings['broker_platform']+')...')
            broker_platform = broker_platform_module(before_broker_platform_loop, after_broker_platform_loop, broker_platform_settings)
            pair = states["latest_broker_plot_settings"]["pair"]
            timezone = states["latest_broker_plot_settings"]["timezone"]
            counts = states["latest_broker_plot_settings"]["counts"]
            if pair.islower():
                pair = translate_pair_to_splited(pair)
            instrument = broker_platform.getInstrument(pair)
            print('')
            print('Ploting...')
            plotter.plot_candles(pair+' in '+timezone, instrument.getRecent1MCandles(counts), timezone)
            print('Ploted.')

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
            plot_historical_pair_data(pair, from_timezone.localize(datetime(from_year, from_month, from_day, 0, 0)), to_timezone.localize(datetime(to_year, to_month, to_day, 0, 0)+time_delta_1_days), timezone)
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
            plot_historical_pair_data(pair, from_timezone.localize(datetime(from_year, from_month, from_day, 0, 0)), to_timezone.localize(datetime(to_year, to_month, to_day, 0, 0)+time_delta_1_days), timezone)
            print('Ploted.')

        elif answer == 12:
            progress_dialog(
                title="Updating historical data",
                text= datetime.now().strftime("Today's date in your timezone: %Y,%m,%d."),
                run_callback=update_historical_pair_data,
            ).run()
            states['latest_historical_pair_data_update'] = datetime.now().strftime("%Y,%m,%d, %H:%M:%S")
            save_states()

        elif answer == 20:
            nsp_selections = [(filename, filename) for filename in os.listdir('./coin_wizard/notification_service_providers')]
            nsp_answer = radiolist_dialog(title='Notification service provider', text='What do you want to do? \nNotification service provider(Current: "'+ settings['notification_service_provider'] +'"). \n', values = nsp_selections).run()
            if nsp_answer == None:
                continue
            settings['notification_service_provider'] = nsp_answer
            notification_service_provider_module = __import__(settings['notification_service_provider']).NotificationServiceProvider
            if settings['notification_service_provider'] not in states["notification_service_provider_settings_dict"]:
                states["notification_service_provider_settings_dict"][nsp_answer] = {}
                save_states()
            save_settings()

        elif answer == 21:
            nsp_name = settings['notification_service_provider']
            nsp_settings = {}
            print('\n=== "Notification service provider(Current: "'+nsp_name+'")" settings ===')
            # print(notification_service_provider_module)
            for field in notification_service_provider_module.notification_service_provider_settings_fields:
                nsp_settings[field] = session.prompt("    "+field + ": ")
            states["notification_service_provider_settings_dict"][nsp_name] = nsp_settings
            save_states()
        else:
            print('Good bye!')
            break
