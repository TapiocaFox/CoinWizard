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
#
# class CoinNet(nn.Module):
#     def __init__(self, in_dim=14, lstm_dim=256, mlp_dim=256, dropout=0., num_actions=3, critic=False):
#         super().__init__()
#
#         self.critic = critic
#         self.linear_pre_lstm = nn.Sequential(
#             nn.Linear(in_dim, mlp_dim),
#             nn.ELU(),
#             nn.Linear(mlp_dim, mlp_dim),
#             nn.ELU()
#         )
#         self.lstm = nn.LSTM(mlp_dim, lstm_dim, 1, batch_first=True)
#
#         if critic:
#             self.mlp_head = nn.Linear(lstm_dim, 1)
#         else:
#             # For later dot product
#             self.mlp_head = nn.Sequential(
#                 nn.Linear(lstm_dim, num_actions),
#                 nn.Softmax(dim=-1)
#             )
#
#     def forward(self, x):
#         self.lstm.flatten_parameters()
#         x = self.linear_pre_lstm(x)
#         lstm_out, hs = self.lstm(x)
#
#         # Batch Size, Length, Features
#         x = lstm_out[:, -1, :]
#
#         x = self.mlp_head(x)
#         return x

class CoinNet(nn.Module):
    def __init__(self, in_dim=14, lstm_dim=256, mlp_dim=256, dropout=0., num_actions=3, critic=True, q_net=False):
        super().__init__()
        self.lstm_dim = lstm_dim

        self.critic = critic

        self.linear_pre_lstm = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ELU()
        )
        self.lstm = nn.LSTM(mlp_dim, lstm_dim, 1, batch_first=True)

        if q_net:
            self.mlp_head = nn.Linear(lstm_dim, num_actions)
            
        elif critic:
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

    def initialize(self):

        for value in self.lstm.state_dict():
            #format values
            param = self.lstm.state_dict()[value]
            print(value)
            if 'weight_ih' in value:
                #print(value,param.shape,'Orthogonal')
                torch.nn.init.orthogonal_(self.lstm.state_dict()[value])#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio

            elif 'weight_hh' in value:
                #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM
                weight_hh_data_ii = torch.eye(self.lstm_dim, self.lstm_dim)#H_Wii
                weight_hh_data_if = torch.eye(self.lstm_dim, self.lstm_dim)#H_Wif
                weight_hh_data_ic = torch.eye(self.lstm_dim, self.lstm_dim)#H_Wic
                weight_hh_data_io = torch.eye(self.lstm_dim, self.lstm_dim)#H_Wio
                weight_hh_data = torch.stack([weight_hh_data_ii, weight_hh_data_if, weight_hh_data_ic, weight_hh_data_io], dim=0)
                weight_hh_data = weight_hh_data.view(self.lstm_dim*4,self.lstm_dim)
                #print(value,param.shape,weight_hh_data.shape,self.number_of_layers,self.lstm_dim,'Identity')
                self.lstm.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY

            elif 'bias' in value:
                #print(value,param.shape,'Zeros')
                torch.nn.init.constant_(self.lstm.state_dict()[value], val=0)
                self.lstm.state_dict()[value].data[self.lstm_dim:self.lstm_dim*2].fill_(1)#set the forget gate | (b_ii|b_if|b_ig|b_io)

        for value in self.linear_pre_lstm.state_dict():
            if 'weight' in value:
                torch.nn.init.eye_(self.linear_pre_lstm.state_dict()[value])

        for value in self.mlp_head.state_dict():
            if 'weight' in value:
                torch.nn.init.normal_(self.mlp_head.state_dict()[value], 0, 0.001)
