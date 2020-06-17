import torch
import torch.nn as nn


class RetraceLoss(nn.Module):
    """
    Retrace Loss as defined in
    Munos, Rémi, Stepleton, Tom, Harutyunyan, Anna, and Bellemare, Marc G.
    Safe and efficient off-policy reinforcement learning.
    In Neural Information Processing Systems (NIPS), 2016.
    """

    def __init__(self,
                 gamma=0.95):
        super().__init__()
        self.gamma = gamma
        self.distance = nn.MSELoss()

    def forward(self,
                state_trajectory_action_values,
                target_state_trajectory_action_values,
                target_state_current_action_values,
                rewards,
                original_log_trajectory_action_probs,
                target_log_trajectory_task_action_probs):
        # TODO: Figure out what the expectation in the temporal difference definition is supposed to be.
        num_steps = state_trajectory_action_values.shape[0]
        log_importance_weights = target_log_trajectory_task_action_probs - original_log_trajectory_action_probs
        log_importance_weights = torch.clamp(log_importance_weights, min=0)
        retrace_state_action_values = torch.zeros_like(target_state_trajectory_action_values)
        # TODO: Implement without nested loops.
        for i in range(num_steps):
            for j in range(i, num_steps):
                importance = self.gamma ** (j - i) * torch.exp(log_importance_weights[i:j, :].sum(0))
                reward = rewards[j, :]
                temporal_difference = (target_state_current_action_values[i, :]
                                       - target_state_trajectory_action_values[j, :])
                retrace_state_action_values[i, :] += importance * (reward + temporal_difference)

        return self.distance(state_trajectory_action_values, retrace_state_action_values)






