#!/usr/bin/python3

import pytz, torch, math, os
from datetime import datetime, timedelta

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trading_agents.dc011.nets as nets
import trading_agents.dc011.dataloader as dataloader
import matplotlib.pyplot as plt
from .timefeatures import time_features
import coin_wizard.historical_pair_data as hist

from coin_wizard.technical_indicators import TechnicalIndicators

utc = pytz.utc

### Global settings ###
verbose = False
stance_dict = {
    0: 'long ',
    1: 'short',
    2: 'wait ',
}
# granularities = ['M5', 'M15', 'H1', 'H4']
# granularities = ['M5', 'M15', 'H1', 'H4', 'D']
granularities = ['M15', 'H1', 'H4']
macd_amplifier = [8, 4, 2]
# granularities = ['M5', 'M15', 'H1', 'H4', 'D']

granularity_time_delta = {
    "M1": timedelta(seconds=60),
    "M5": timedelta(seconds=60*5),
    "M15": timedelta(seconds=60*15),
    "M30": timedelta(seconds=60*30),
    "H1": timedelta(seconds=60*60),
    "H4": timedelta(seconds=60*240),
    "D": timedelta(seconds=60*60*24),
}
# input_period_list = [64, 64, 64]
# input_period_list = [64, 64, 64, 64]
# input_period_list = [96, 96, 96, 96]
# input_period_list = [64, 64, 64, 64, 32]
# input_period_list = [96, 128, 96, 96, 96, 32]
# input_period_list = [256, 256, 256]
input_period_list = [288, 288, 288]
# input_period_list = [384, 384, 384]
# input_period_list = [480, 480, 480]

cci_trigger_granularity = 0
decode_inference_granularity = 2
# decode_inference_len = 64
decode_inference_len = 72
decode_prediction_granularity = 0
decode_predict_len = 48
# decode_predict_len = 24
decode_predict_resolution = 24
lamb = np.log(1-0.5)/np.log(np.exp(-decode_predict_len))
cuda_enabled = True
# cuda_enabled = False
selected_net = 'backup.net'
# selected_net = 'DC_2021_09_05_23_13.net'
plot_range = 0.0026

### Indicator settings ###
# moving_average = 9
momentum_period=10
rsi_period=14
cci_period=14

### Train settings ###
load_selected_net = True
learning_rate = 0.00001
fine_tune = True
# learning_rate = learning_rate * 0.23960349927
primary_intrument = 'eurusd'
primary_intrument_trade = 'EUR_USD'
primary_intrument_index = 0
# primary_intrument = 'usdchf'
# primary_intrument_trade = 'USD_CHF'
# primary_intrument_index = 9
# selected_instrument_list = ['eurusd', 'gbpusd', 'audusd', 'usdcad', 'eurjpy', 'usdjpy']

# selected_instrument_datascaler_list = [ dataloader.DataScaler(np.array(1.1924490551226474), np.array(0.09608276821936781)),
#                                         dataloader.DataScaler(np.array(1.428601840627313), np.array(0.14754311756904043)),
#                                         dataloader.DataScaler(np.array(0.8109898637830456), np.array(0.12188596874201743)),
#                                         dataloader.DataScaler(np.array(1.222725673225676), np.array(0.13292115495331883)),
#                                         dataloader.DataScaler(np.array(125.33177145959843), np.array(10.895352856678667)),
#                                         dataloader.DataScaler(np.array(105.68095956301106), np.array(11.318548392932305))]

selected_instrument_list = ['eurusd', 'gbpusd', 'audusd', 'eurjpy', 'nzdusd',
                            'usdcad', 'usdjpy', 'gbpaud', 'euraud', 'gbpcad',
                            'gbpnzd', 'nzdchf', 'cadchf', 'eurcad', 'gbpchf',
                            'audjpy', 'eurnok', 'usdtry', 'audnzd', 'audchf',
                            'sgdjpy', 'xagusd', 'xauusd', 'zarjpy', 'usdzar',

                            'gbpjpy', 'usdczk', 'audcad', 'cadjpy', 'chfjpy',
                            'eurgbp', 'usdnok', 'xauaud', 'xaugbp', 'xaueur',
                            'eurczk', 'nzdcad', 'usdsgd', 'usdchf', 'eurtry',
                            ]

selected_instrument_datascaler_list = [
                                            dataloader.DataScaler(np.array(1.2226727737949628), np.array(0.0015582402708278174), (np.array(0.8245795527757817), np.array(0.0010407336554503633))),
                                            dataloader.DataScaler(np.array(1.45501340226881), np.array(0.0018410780600621854), (np.array(0.6944530971657596), np.array(0.000897687910107338))),
                                            dataloader.DataScaler(np.array(0.8409227146528752), np.array(0.0013779465889715195), (np.array(1.2173862015744954), np.array(0.001985033987049325))),
                                            dataloader.DataScaler(np.array(123.22119330754076), np.array(0.18733760914037093), (np.array(0.008186416107584096), np.array(1.2785196787446727e-05))),
                                            dataloader.DataScaler(np.array(0.7346609054141837), np.array(0.0012690734157646652), (np.array(1.3730623095712402), np.array(0.0023793789604519824))),

                                            dataloader.DataScaler(np.array(1.1841049275904305), np.array(0.001393785442655719), (np.array(0.8579560513445401), np.array(0.0010185741021392022))),
                                            dataloader.DataScaler(np.array(101.70687582712374), np.array(0.1280890673058432), (np.array(0.010023415791657853), np.array(1.2729417314643028e-05))),
                                            dataloader.DataScaler(np.array(1.7489657289994534), np.array(0.0027142330190730547), (np.array(0.5766111839612996), np.array(0.0008861522209003131))),
                                            dataloader.DataScaler(np.array(1.4704677095197058), np.array(0.002112831837152215), (np.array(0.6851559784596104), np.array(0.0009821517239837786))),
                                            dataloader.DataScaler(np.array(1.7050919335627375), np.array(0.0022733263216393822), (np.array(0.5894682775027766), np.array(0.0007849350028570728))),

                                            dataloader.DataScaler(np.array(1.982708837715431), np.array(0.003268181944023001), (np.array(0.5065921940839548), np.array(0.0008249428984395599))),
                                            dataloader.DataScaler(np.array(0.7022315989619157), np.array(0.0011926923591010603), (np.array(1.4323212856562635), np.array(0.0024494270546285674))),
                                            dataloader.DataScaler(np.array(0.8205161447191238), np.array(0.0011933777762369428), (np.array(1.235808265135375), np.array(0.001775612282246314))),
                                            dataloader.DataScaler(np.array(1.433110284870101), np.array(0.0018295687141570555), (np.array(0.7002372294521236), np.array(0.0008987075010232095))),
                                            dataloader.DataScaler(np.array(1.3918337555225828), np.array(0.0018957498566576404), (np.array(0.7246585175431945), np.array(0.0009914910750076992))),

                                            dataloader.DataScaler(np.array(84.04944948208193), np.array(0.15862065209357476), (np.array(0.011988386059928918), np.array(2.3033473135028014e-05))),
                                            dataloader.DataScaler(np.array(8.854169837315494), np.array(0.01078380023386487), (np.array(0.11441015554546918), np.array(0.00013404402277801653))),
                                            dataloader.DataScaler(np.array(3.42330615833745), np.array(0.006855786304454206), (np.array(0.36742037340075556), np.array(0.0006249576373701751))),
                                            dataloader.DataScaler(np.array(1.1399203007488543), np.array(0.001232440516230078), (np.array(0.8834800696059951), np.array(0.0009485977308032728))),
                                            dataloader.DataScaler(np.array(0.8033054645248916), np.array(0.0013308023296079174), (np.array(1.2689265265399547), np.array(0.0021012245214045134))),

                                            dataloader.DataScaler(np.array(76.38232800119137), np.array(0.10013714825059432), (np.array(0.013265018486138017), np.array(1.803955839518167e-05))),
                                            dataloader.DataScaler(np.array(21.14781348972715), np.array(0.09095701770204394), (np.array(0.051400577718606504), np.array(0.00019655292246843287))),
                                            dataloader.DataScaler(np.array(1389.8507780231075), np.array(3.1098342142082243), (np.array(0.0007356900144239168), np.array(1.6037441284644798e-06))),
                                            dataloader.DataScaler(np.array(8.852639836823544), np.array(0.023127164682104996), (np.array(0.11628637971154769), np.array(0.00031377956361584126))),
                                            dataloader.DataScaler(np.array(12.073743261105015), np.array(0.028976414893401654), (np.array(0.08882246487819616), np.array(0.00020644898537478375))),

                                            dataloader.DataScaler(np.array(146.94883903543672), np.array(0.24138156009816492), (np.array(0.006905001306824298), np.array(1.154016358952686e-05))),
                                            dataloader.DataScaler(np.array(21.73129795531456), np.array(0.031955581139748), (np.array(0.04658759402623105), np.array(6.991211397683505e-05))),
                                            dataloader.DataScaler(np.array(0.9773231643543236), np.array(0.0012336168367192166), (np.array(1.0252740581476445), np.array(0.00130321477799189))),
                                            dataloader.DataScaler(np.array(85.99703161581871), np.array(0.14533326332574023), (np.array(0.011695720742257734), np.array(1.9930311892417374e-05))),
                                            dataloader.DataScaler(np.array(106.12622235088332), np.array(0.1505194604840329), (np.array(0.009589886755925096), np.array(1.425809533480594e-05))),

                                            dataloader.DataScaler(np.array(0.842727828980084), np.array(0.0010011659819559372), (np.array(1.191358333333061), np.array(0.0014221490580598507))),
                                            dataloader.DataScaler(np.array(7.347265629033329), np.array(0.012873449871549071), (np.array(0.14107633230771463), np.array(0.000245065577215033))),
                                            dataloader.DataScaler(np.array(1540.9470141007052), np.array(3.5130337186000578), (np.array(0.000654562148216634), np.array(1.503670051191871e-06))),
                                            dataloader.DataScaler(np.array(900.9406697635376), np.array(2.03870964411316), (np.array(0.001129281475217387), np.array(2.559468239382574e-06))),
                                            dataloader.DataScaler(np.array(1081.3723723600124), np.array(2.3237191677445015), (np.array(0.0009371596837398898), np.array(2.0150748934707676e-06))),

                                            dataloader.DataScaler(np.array(26.184868096651602), np.array(0.017247943384006596), (np.array(0.03824319583299129), np.array(2.5482089643169594e-05))),
                                            dataloader.DataScaler(np.array(0.8617517968179974), np.array(0.0012330052814185403), (np.array(1.1669835433001823), np.array(0.001689506861623083))),
                                            dataloader.DataScaler(np.array(1.3287208662521401), np.array(0.0010463199267785335), (np.array(0.754260698899208), np.array(0.0005921438958232278))),
                                            dataloader.DataScaler(np.array(0.9588186652969188), np.array(0.0012430947322487201), (np.array(1.046103656078779), np.array(0.0013698662897521203))),
                                            dataloader.DataScaler(np.array(4.021334735179712), np.array(0.008316761122223485), (np.array(0.2974813826464468), np.array(0.000521362918911068))),
                                        ]
fit_granularity = 'M15'
learning_rate_decay = 0.965**1
epoch_counts = int(512*1) # 6
batch_counts = int(512*2*2)
# batch_counts = int(64)
test_batch_counts = 16
# batch_counts = int(1)
# batch_counts = 32
batch_size = 16
batch_accumulation = 2
training_batches_update_epochs = 4



neighbors = 3
interval = 2

### Running settings ###
loop_every_n_15_second = 4*15
input_additional_period = 64
trade_additional_period = 5
close_steps = decode_predict_len + trade_additional_period
backtesting_plot = True
# backtesting_trading_active_plot = True
backtesting_trade_plot = False
backtesting_plot_pause = 0.001
activate_watchdog_cci_threshold = 100
watchdog_active_period = 6 # neighbors*interval

trade_trigger_ratio = 2.0
# trade_trigger_magnitude = 0.00135
trade_upper_trigger_threshold = 0.5
trade_lower_trigger_threshold = 0.1

cci_close_granularity = 'M15'
cci_close_threshold = 100 * 9999


trailing_stop_distance = 0.0015
risk_reward_ratio = 0.75
take_profit_magnitude = 0.0015
stop_lost_magnitude = 0.00135
rsi_trade_close_rsi = 35
trade_close_price_distance = 0.0003
trade_close_least_steps = 8
trials = 3
trade_cd = 4
do_not_trade = False
# trials = 99
# mode = 'cci'

import coin_wizard.plotter as plotter

ti = TechnicalIndicators()

def calc_confusion_matrix(answers, outputs, labels_length):
    answers = answers.tolist()
    outputs = outputs.tolist()
    array = np.zeros((labels_length, labels_length))
    for i, item in enumerate(answers):
        array[item, outputs[i]] += 1

    # print(array)
    # print(np.sum(array, axis=1))
    array = array / np.sum(array, axis=1)[:, None]
    return array

class TradingAgent(object):
    def __init__(self, agent_directory):
        print(agent_directory)
        if not os.path.exists(os.path.join(agent_directory, 'figs')):
            os.makedirs(os.path.join(agent_directory, 'figs'))
        self.agent_directory = agent_directory
        self.every_15_second_loop_count = 0

        self.nsp = None
        self.net = None
        self.position = 0

        self.current_trade = None
        self.current_trade_total_steps = 0
        self.current_trade_best_profit = 0
        self.current_trade_worst_loss = 0
        self.current_trade_steps = 0
        self.current_trade_close_steps = 0

        self.total_long_counts = 0
        self.total_short_counts = 0

        self.total_long_pl = 0
        self.total_short_pl = 0

        self.succeeded_trials = 0

        self.watchdog_remaining_active_period = 0
        self.trade_cd = 0

    def _order_canceled_listener(self, order, reason):
        self.position = 0

    def _order_filled_listener(self, order, trade):
        # Record stats
        if trade.getTradeSettings()['units'] > 0:
            self.total_long_counts += 1
        else:
            self.total_short_counts += 1

        self.current_trade = trade
        trade.onReduced(self._trade_reduced_listener)
        trade.onClosed(self._trade_closed_listener)

    def _trade_reduced_listener(self, trade, units, realized_pl, close_price, spread, timestamp):
        pass

    def _trade_closed_listener(self, trade, realized_pl, close_price, spread, timestamp):
        # Record stats
        if trade.getTradeSettings()['units'] > 0:
            self.total_long_pl += realized_pl
        else:
            self.total_short_pl += realized_pl

        self.position = 0

        self.nsp.pushImmediately('Trade closed', 'Trade closed after %d total steps. PL: %.5f. Best profit: %.5f. Worst loss: %.5f.' % (
            self.current_trade_steps, realized_pl, self.current_trade_best_profit, self.current_trade_worst_loss))

    def _neural_net_predict(self, mid=None):
        pass

    def _every_15_second_loop(self, BrokerAPI):
        pass

    def run(self, BrokerAPI):
        pass

    def stop_running(self, BrokerAPI):
        pass

    def train(self, BrokerAPI):
        pass

    def stop_training(self, BrokerAPI):
        pass

    def test(self, BacktestBrokerAPI):
        pass

    def stop_testing(self, BacktestBrokerAPI):
        self.stop_running
