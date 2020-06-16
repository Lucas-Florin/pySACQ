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
        # TODO: separate loss for each task head?
        self.distance = nn.MSELoss()

    def forward(self,
                state_action_values,
                target_state_action_values,
                rewards,
                log_trajectory_action_probs,
                log_task_action_probs):
        # TODO: Figure out what the expectation in the temporal difference definition is supposed to be.
        num_steps = state_action_values.shape[0]
        log_importance_weights = log_task_action_probs - log_trajectory_action_probs
        # TODO: use torch.clamp function
        # TODO: Fix: use min instead of max.
        log_importance_weights[log_importance_weights > 0] = 0
        retrace_state_action_values = torch.zeros_like(target_state_action_values)
        # TODO: Implement without nested loops.
        for i in range(num_steps):
            for j in range(i, num_steps):
                importance = self.gamma ** (j - i) * torch.exp(log_importance_weights[i:j, :].sum(0))
                reward = rewards[j, :]
                # TODO: Use state action values for actions sampled with the current policy for the i index.
                temporal_difference = target_state_action_values[i, :] - target_state_action_values[j, :]
                retrace_state_action_values[i, :] += importance * (reward + temporal_difference)

        return self.distance(state_action_values, retrace_state_action_values)






