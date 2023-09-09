import torch, copy, math
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from .pipeline import Pipeline
from typing import List

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0

class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0

class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0

class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0

class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0

class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0

class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0

class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0

features_by_offsets = {
    offsets.YearEnd: [],
    offsets.QuarterEnd: [],
    offsets.MonthEnd: [],
    offsets.Week: [],
    offsets.Day: [DayOfWeek],
    offsets.BusinessDay: [DayOfWeek],
    offsets.Hour: [HourOfDay, DayOfWeek],
    offsets.Minute: [
        MinuteOfHour,
        HourOfDay,
        DayOfWeek
    ],
    offsets.Second: [
        SecondOfMinute,
        MinuteOfHour,
        HourOfDay,
        DayOfWeek
    ],
}

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    # features_by_offsets = {
    #     offsets.YearEnd: [],
    #     offsets.QuarterEnd: [MonthOfYear],
    #     offsets.MonthEnd: [MonthOfYear],
    #     offsets.Week: [DayOfMonth, WeekOfYear],
    #     offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
    #     offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
    #     offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    #     offsets.Minute: [
    #         MinuteOfHour,
    #         HourOfDay,
    #         DayOfWeek,
    #         DayOfMonth,
    #         DayOfYear,
    #     ],
    #     offsets.Second: [
    #         SecondOfMinute,
    #         MinuteOfHour,
    #         HourOfDay,
    #         DayOfWeek,
    #         DayOfMonth,
    #         DayOfYear,
    #     ],
    # }


    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates_series, device, freq='T'):
    dates = pd.to_datetime(dates_series.values)
    result_list = []
    for feat in time_features_from_frequency_str(freq):
        t = 2*math.pi*torch.tensor(feat(dates).to_list(), device=device)
        result_list.append(torch.sin(t))
        result_list.append(torch.cos(t))

    return torch.stack(result_list).transpose(1, 0)

def get_freq_features(freq_str):
    offset = to_offset(freq_str)
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return feature_classes
    return None

class AppendTimeFeaturesPipline(Pipeline):
    def __init__(self, label='timestamp', freq='T'):
        super().__init__(torch.Tensor, torch.Tensor)
        self.label = label
        self.freq = freq
        self.freq_features = get_freq_features(freq)

    def setupInSpecs(self, in_specs):
        self.in_specs = in_specs
        self.device = in_specs['device']
        self.out_specs = copy.deepcopy(self.in_specs)
        for feature in self.freq_features:
            self.out_specs['tensor_feature_description_list'].append({'type': 'time_feature', 'feature': feature.__name__, 'sin_cos': 'sin'})
        # for feature in self.freq_features:
            self.out_specs['tensor_feature_description_list'].append({'type': 'time_feature', 'feature': feature.__name__, 'sin_cos': 'cos'})

    def generateOutSpecs(self):
        return self.out_specs

    def _process(self, in_data, attachment_dict):
        timestamp_series = attachment_dict['ohlc_df'][self.label]
        time_features_tensor = time_features(timestamp_series, self.device, self.freq)
        # print(time_features_tensor)
        return torch.cat((in_data, time_features_tensor), dim=-1), attachment_dict
