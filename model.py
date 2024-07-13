import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, input_size):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 128, num_layers=2, batch_first=True, dropout=0.1
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # output between -1 and 1
        )
        self.actor_std = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mean = self.actor_mean(lstm_out[:, -1, :])
        std = self.actor_std(lstm_out[:, -1, :]) + 1e-6
        value = self.critic(lstm_out[:, -1, :])
        return mean, std, value

    def get_action(self, state):
        mean, std, value = self(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
