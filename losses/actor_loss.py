import torch
import torch.nn as nn


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_action_values, log_likelihoods, alpha=1.0):
        return - (state_action_values + alpha * log_likelihoods)



