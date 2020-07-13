import torch





class FetchTransform(torch.nn.Module):

    def forward(self, observation):
        return torch.cat([torch.tensor(t, dtype=torch.float) for t in [observation['observation'], observation['desired_goal']]], dim=0)

