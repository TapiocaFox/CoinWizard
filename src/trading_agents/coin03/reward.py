import torch

class RewardCalculator(object):
    def __init__(self, trading_amount_rate=1, spread=0.0002, cuda=False):

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

        # Settings
        self.trading_amount_rate = trading_amount_rate
        self.spread = spread/2
        # Imported values
        self.granularity_list = []
        self.episode_candles_fast_inference_index_list = []
        self.primary_granularity = None
        self.episode_steps = None

        # Runtime values
        self.open_price_tensor = None
        self.accumulated_trade_steps_tensor = None
        self.total_log_returns_tensor = None
        self.total_log_returns_without_spread_tensor = None
        self.total_long_log_returns_tensor = None
        self.total_short_log_returns_tensor = None
        self.long_counts_tensor = None
        self.long_steps_tensor = None
        self.short_counts_tensor = None
        self.short_steps_tensor = None
        self.position_state_tensor = None
        self.concurrent_episodes = None

        self.short_position = -1
        self.wait_position = 0
        self.long_position = 1

        # Pre calculated values
        self.all_actions_tensor = None

    def reset(self, concurrent_episodes, granularity_list, episode_candles_fast_inference_index_list, primary_granularity, episode_steps):
        close_index = 3
        wait_position = 1

        self.concurrent_episodes = concurrent_episodes

        self.granularity_list = granularity_list
        self.episode_candles_fast_inference_index_list = episode_candles_fast_inference_index_list
        self.primary_granularity = primary_granularity
        self.primary_close_tensor = torch.stack([candles[primary_granularity][fast_inference_index_list[0][primary_granularity]:fast_inference_index_list[-1][primary_granularity]+1, close_index]
                                            for candles, fast_inference_index_list in self.episode_candles_fast_inference_index_list])
        self.primary_close_diff_tensor = self.primary_close_tensor[:, 1:] - self.primary_close_tensor[:, :-1]
        self.episode_steps = episode_steps

        self.open_price_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.accumulated_trade_steps_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.total_log_returns_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.total_log_returns_without_spread_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.total_long_log_returns_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.total_short_log_returns_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.position_state_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.long_counts_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.long_steps_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.short_counts_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.short_steps_tensor = torch.zeros((self.concurrent_episodes), device=self.device)
        self.all_actions_tensor = torch.tensor([-1, 0, 1], device=self.device).unsqueeze(0).expand(self.concurrent_episodes, -1)

    def step(self, current_episode_steps, actions, rewards_for_all_actions=True):
        old_actions = self.position_state_tensor
        done = current_episode_steps >= self.episode_steps
        trading_amount_rate = self.trading_amount_rate
        spread = self.spread
        price_diff = self.primary_close_diff_tensor[:, current_episode_steps-1]

        # Calculate single action's reward
        actions = actions - 1 # translate offest 0, 1, 2 -> -1, 0, 1
        spread_terms = trading_amount_rate*torch.abs(actions-old_actions)*spread
        v = 1.0 + trading_amount_rate*actions*price_diff - spread_terms
        reward = torch.log(v)

        # spread_terms = torch.abs(actions-old_actions)*spread
        # v = actions*price_diff - spread_terms
        # reward = v

        self.total_log_returns_tensor += reward
        self.total_log_returns_without_spread_tensor += torch.log(1.0 + trading_amount_rate*actions*price_diff)
        # self.total_log_returns_without_spread_tensor += actions*price_diff

        now_short_tensor = (actions==self.short_position)
        now_wait_tensor = (actions==self.wait_position)
        now_long_tensor = (actions==self.long_position)

        prev_short_tensor = (old_actions==self.short_position)
        prev_wait_tensor = (old_actions==self.wait_position)
        prev_long_tensor = (old_actions==self.long_position)

        self.short_counts_tensor += torch.logical_and(prev_wait_tensor, now_short_tensor).int()
        self.short_steps_tensor += now_short_tensor.int()
        reward_filter = torch.logical_or(prev_short_tensor, now_short_tensor)
        self.total_short_log_returns_tensor[reward_filter] += reward[reward_filter]

        self.long_counts_tensor += torch.logical_and(prev_wait_tensor, now_long_tensor).int()
        self.long_steps_tensor += now_long_tensor.int()
        reward_filter = torch.logical_or(prev_long_tensor, now_long_tensor)
        self.total_long_log_returns_tensor[reward_filter] += reward[reward_filter]

        self.position_state_tensor = actions.clone()

        if rewards_for_all_actions:
            spread_terms = trading_amount_rate*torch.abs(self.all_actions_tensor-old_actions.unsqueeze(1))*spread
            v = 1.0 + trading_amount_rate*self.all_actions_tensor*price_diff.unsqueeze(1) - spread_terms
            # spread_terms = torch.abs(self.all_actions_tensor-old_actions.unsqueeze(1))*spread
            # v = self.all_actions_tensor*price_diff.unsqueeze(1) - spread_terms
            return torch.log(v)

        else:
            return reward

    def get_summary(self):

        return {
            # 'idle_punishment_baseline': self.idle_punishment_ratio*self.episode_steps,

            'total_reward_average': self.total_log_returns_tensor.mean().item(),

            'total_log_returns_average': self.total_log_returns_tensor.mean().item(),
            'total_log_returns_tensor': self.total_log_returns_tensor.tolist(),
            'total_log_returns_without_spread_tensor': self.total_log_returns_without_spread_tensor.tolist(),

            'total_long_log_returns_tensor': self.total_long_log_returns_tensor.tolist(),
            'long_counts_tensor': self.long_counts_tensor.tolist(),
            'long_steps_tensor': self.long_steps_tensor.tolist(),
            'long_mean_steps_tensor': (self.long_steps_tensor.sum()/(self.long_counts_tensor.sum()+1e-10)).item(),

            'total_short_log_returns_tensor': self.total_short_log_returns_tensor.tolist(),
            'short_counts_tensor': self.short_counts_tensor.tolist(),
            'short_steps_tensor': self.short_steps_tensor.tolist(),
            'short_mean_steps_tensor': (self.short_steps_tensor.sum()/(self.short_counts_tensor.sum()+1e-10)).item(),
        }
