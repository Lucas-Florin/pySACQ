import torch
import torch.nn as nn


class RetraceLossRecursive(nn.Module):
    """
    Retrace Loss as defined in
    Munos, Rémi, Stepleton, Tom, Harutyunyan, Anna, and Bellemare, Marc G.
    Safe and efficient off-policy reinforcement learning.
    In Neural Information Processing Systems (NIPS), 2016.
    """

    def __init__(self,
                 gamma=0.95, use_gpu=False):
        super().__init__()
        self.gamma = gamma
        self.distance = nn.SmoothL1Loss()
        self.use_gpu = use_gpu

    def forward(self,
                state_trajectory_action_values,
                target_state_trajectory_action_values,
                target_state_current_action_values,
                rewards,
                original_log_trajectory_action_probs,
                target_log_trajectory_task_action_probs):

        num_steps = state_trajectory_action_values.shape[0]
        num_intentions = state_trajectory_action_values.shape[1]
        log_importance_weights = target_log_trajectory_task_action_probs - original_log_trajectory_action_probs
        importance_weights = torch.exp(torch.clamp(log_importance_weights, max=0))
        padding = torch.zeros((1, num_intentions))
        padding = padding.cuda() if self.use_gpu else padding
        target_state_current_action_values = torch.cat([target_state_current_action_values[1:, :],
                                                        padding], dim=0)
        temporal_difference = (self.gamma * target_state_current_action_values
                               - target_state_trajectory_action_values)
        temporal_difference = rewards + temporal_difference
        retrace_state_action_values = torch.zeros_like(state_trajectory_action_values)

        retrace_state_action_values[-1, :] = importance_weights[-1, :] * temporal_difference[-1, :]
        for j in range(num_steps - 2, -1, -1):
            retrace_state_action_values[j, :] = (importance_weights[j, :] * temporal_difference[j, :]
                                                 + (self.gamma * importance_weights[j + 1, :]
                                                    * temporal_difference[j + 1, :]))

        loss = self.distance(state_trajectory_action_values, retrace_state_action_values)
        return loss





