import torch
import torch.nn.functional as F


class RetraceFast(torch.nn.Module):
    def __init__(self, gamma=0.99):
        super(RetraceFast, self).__init__()
        self.gamma = gamma

    def forward(self,
                Q,
                expected_target_Q,
                target_Q,
                rewards,
                target_policy_probs,
                behaviour_policy_probs):
        """
        Implementation of Retrace loss ((http://arxiv.org/abs/1606.02647)) in PyTorch.

        Args:
            Q: State-Action values.
            Torch tensor with shape `[B, (T+1)]`

            expected_target_Q: 𝔼_π Q(s_t,.) (from the fixed critic network)
            Torch tensor with shape `[B, (T+1)]`

            target_Q: State-Action values from target network.
            Torch tensor with shape `[B, (T+1)]`

            rewards: Holds rewards for taking an action in the environment.
            Torch tensor with shape `[B, (T+1)]`

            target_policy_probs: Probability of target policy π(a|s)
            Torch tensor with shape `[B, (T+1)]`

            behaviour_policy_probs: Probability of behaviour policy b(a|s)
            Torch tensor with shape `[B, (T+1)]`

            gamma: Discount factor

            recursive: If true, uses the recursive equation for retrace otherwise iterative. For more information see
            the docstrings.

        Returns:
            Retrace loss
        """



        """
        For information on the parameters see class docs.

        Computes the retrace loss recursively according to
        L = 𝔼_τ[(Q_t - Q_ret_t)^2]
        Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1

        with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}

        Returns:
            Scalar critic loss value.

        """

        Q = remove_last_timestep(Q)

        with torch.no_grad():
            log_importance_weights = (target_policy_probs - behaviour_policy_probs.unsqueeze(-1))
            log_importance_weights = torch.clamp(log_importance_weights, max=0)
            log_importance_weights = remove_first_timestep(log_importance_weights)
            importance_weights = (torch.exp(log_importance_weights))
            expected_target_Q = remove_first_timestep(expected_target_Q)
            target_Q = remove_first_timestep(target_Q)
            rewards = remove_last_timestep(rewards)

            td = rewards + self.gamma * (expected_target_Q - importance_weights * target_Q)
            td[:, -1, :] = rewards[:, -1, :] + self.gamma * expected_target_Q[:, -1, :]
            decay = torch.cumsum(torch.log(torch.tensor(self.gamma, dtype=torch.float)).to(log_importance_weights.device)
                                 + log_importance_weights, dim=-2)
            decay = torch.exp(torch.clamp(decay, min=-20))

            Q_ret = torch.flip(torch.cumsum(torch.flip(td * decay, [-2]), dim=-2), [-2]) / decay

        return F.mse_loss(Q, Q_ret)


def remove_last_timestep(x):
    return x.narrow(-2, 0, x.shape[-2] - 1)


def remove_first_timestep(x):
    return x.narrow(-2, 1, x.shape[-2] - 1)
