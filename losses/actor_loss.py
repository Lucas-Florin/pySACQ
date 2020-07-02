import torch
import torch.nn as nn


class ActorLoss(nn.Module):
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, state_action_values, log_likelihoods):
        return (-state_action_values + self.alpha * log_likelihoods).mean()



