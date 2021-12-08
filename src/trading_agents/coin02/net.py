import torch
from torch import nn

# class CoinLSTM(nn.Module):
# 	def __init__(self, state_size, action_size):
# 		super(CoinLSTM, self).__init__()
# 		self.first_two_layers = nn.Sequential(
# 			nn.Linear(state_size, 256),
# 			nn.ELU(),
# 			nn.Linear(256, 256),
# 			nn.ELU()
# 		)
# 		self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
# 		self.last_linear = nn.Linear(256, 3)
#
# # Data Flow Protocol:
# # 1. network input shape: (batch_size, seq_length, num_features)
# # 2. LSTM output shape: (batch_size, seq_length, hidden_size)
# # 3. Linear input shape:  (batch_size * seq_length, hidden_size)
# # 4. Linear output: (batch_size * seq_length, out_size)
#
# 	def forward(self, input):
# 		# rint(input.size())
# 		x = self.first_two_layers(input)
# 		# print(x.size())
#
# 		lstm_out, hs = self.lstm(x)
# 		# print(lstm_out.size())
#
# 		batch_size, seq_len, mid_dim = lstm_out.shape
# 		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
# 		# linear_in = lstm_out.contiguous().view(-1, lstm_out.size(2))
#
# 		# linear_in = lstm_out.reshape(-1, hidden_size)
# 		return self.last_linear(linear_in)

class CoinNet(nn.Module):
    def __init__(self, in_dim=14, lstm_dim=256, mlp_dim=256, dropout=0., num_actions=3, critic=False):
        super().__init__()

        self.critic = critic
        self.linear_pre_lstm = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ELU()
        )
        self.lstm = nn.LSTM(mlp_dim, lstm_dim, 1, batch_first=True)

        if critic:
            self.mlp_head = nn.Linear(lstm_dim, 1)
        else:
            # For later dot product
            self.mlp_head = nn.Sequential(
                nn.Linear(lstm_dim, num_actions),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.linear_pre_lstm(x)
        lstm_out, hs = self.lstm(x)

        # Batch Size, Length, Features
        x = lstm_out[:, -1, :]

        x = self.mlp_head(x)
        return x
