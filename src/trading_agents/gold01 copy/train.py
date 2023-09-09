
import torch
from utils import ConfigLoader, calculate_gamma, plot_list, pretty_print, plot_tensor

# Pipelines
from pipelines.printer import PrinterPipeline
from pipelines.initialize import InitializePipeline
from pipelines.append_time_features import AppendTimeFeaturesPipline
from pipelines.append_ohlc import AppendOhlcPipeline
from pipelines.append_position import AppendPositionPipeline
from pipelines.remove_type import RemoveTypePipeline
from pipelines.transform_gaf import TransformGafPipeline
from pipeline_array import PipelineArray

# Gym related
from gyms.hist_data_gym import HistDataGym
from reward_calculators.realistic import RealisticRewardCalculator
from account_book import AccountBook

# Load Configs
config = ConfigLoader('config.json')
ppo_agent_config = ConfigLoader('ppo_agent_config.json')
training_config = ConfigLoader('training_config.json')

# Generic config
cuda = config.get('cuda')
device = torch.device('cuda:0') if cuda else torch.device('cpu')
selected_instrument_list = config.get('selected_instrument_list')
primary_intrument = config.get('primary_intrument')
granularity = config.get('granularity')
observation_period = config.get('observation_period')
net_input_length = config.get('net_input_length')

# PPO agent config


# Training config
episode_steps = training_config.get('episode_steps')
episodes_rounds = training_config.get('episodes_rounds')
primary_instrument_spread = training_config.get('primary_instrument_spread')
train_from_datetime = training_config.getDate('train_start_year', 'train_start_month', 'train_start_day', 'start_hour', 'start_min')
train_to_datetime = training_config.getDate('train_end_year', 'train_end_month', 'train_end_day', 'end_hour', 'end_min')
test_from_datetime = training_config.getDate('test_start_year', 'test_start_month', 'test_start_day', 'start_hour', 'start_min')
test_to_datetime = training_config.getDate('test_end_year', 'test_end_month', 'test_end_day', 'end_hour', 'end_min')

# Setup gym
hist_data_gym = HistDataGym(
                    selected_instrument_list,
                    primary_intrument, granularity,
                    observation_period, train_from_datetime,
                    train_to_datetime,
                    episode_steps,
                    RealisticRewardCalculator(),
                    AccountBook()
                )
gym_specs = hist_data_gym.generateOutSpecs()
gym_specs['net_input_length'] = net_input_length

# Setup piplines
pa = PipelineArray([
    InitializePipeline(device),
    AppendTimeFeaturesPipline(),
    AppendPositionPipeline(200),
    PrinterPipeline(torch.Tensor),
    AppendOhlcPipeline(),
    RemoveTypePipeline('ohlc'),
    TransformGafPipeline(method='s', features_to_channels=True),
    ], gym_specs)

def on_finished_callback():
    pa.resetAll()

def on_step_callback(finished, state):
    state, attachment_dict = pa.process(state)
    # print(state)
    # print(state.shape)
    pass

hist_data_gym.onStep(on_step_callback)
hist_data_gym.onFinished(on_finished_callback)
hist_data_gym.start()
