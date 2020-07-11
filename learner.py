# Import Python packages
import random
import copy
import time

# Import libraries
import torch
import torch.nn as nn

# Import losses
from losses.retrace_loss_recursive import RetraceLossRecursive
from losses.retrace_loss_recursive_fast import RetraceLossRecursiveFast
from losses.retrace_loss import Retrace
from losses.actor_loss import ActorLoss


class Learner:

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 replay_buffer: list,
                 num_learning_iterations=10,
                 episode_batch_size=32,
                 expectation_sample_size=8,
                 lr=0.0002,
                 use_gpu=False,
                 continuous=False,
                 writer=None):

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.num_learning_iterations = num_learning_iterations
        self.episode_batch_size = episode_batch_size
        self.lr = lr
        self.writer = writer
        self.num_intentions = self.actor.num_intentions
        self.expectation_sample_size = expectation_sample_size

        self.use_gpu = use_gpu
        self.continuous = continuous
        self.max_grad_norm = 10.0
        self.step_counter = 0

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr)

        # TODO: Specify alpha and gamma parameters.
        self.actor_criterion = ActorLoss()
        self.critic_criterion = RetraceLossRecursive(use_gpu=use_gpu)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor.eval()
        self.target_critic.eval()

    def update_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

    def get_critic_input(self, actions, states) -> torch.Tensor:
        if self.continuous:
            return self.get_critic_input_continuous(actions, states)
        else:
            return self.get_critic_input_discrete(actions, states)

    def get_critic_input_continuous(self, actions, states) -> torch.Tensor:
        if actions.dim() == 4 and states.dim() == 3:
            states = states.unsqueeze(-2).expand(*([-1] * (actions.dim() - 2) + [self.num_intentions, -1]))

        assert actions.dim() == states.dim()
        critic_input = torch.cat([actions, states], dim=-1)
        return critic_input

    def get_critic_input_discrete(self, actions, states) -> torch.Tensor:
        assert actions.dim() == 2
        assert states.dim() == 2
        if actions.shape[1] == 1:
            critic_input = torch.cat([actions, states], dim=1)
        else:
            critic_input = torch.cat([actions.float().unsqueeze(2),
                                      states.unsqueeze(1).expand(-1, self.num_intentions, -1)], dim=2)
        return critic_input

    def expand_actions(self, actions):
        if self.continuous:
            return actions.unsqueeze(-2).expand(-1, -1, self.actor.num_intentions, -1)
        else:
            assert False
            return actions.expand(-1, self.actor.num_intentions)

    def get_batch(self, size=1):
        """
        returns -- (states, actions, log_probs, rewards)
        """
        trajectories = random.choices(self.replay_buffer, k=size)
        tensors = [
            torch.stack([
                trajectory[tensor]
                for trajectory in trajectories
            ])
            for tensor in range(4)
        ]
        return tuple([t.cuda() if self.use_gpu else t for t in tensors])

    def learn(self):
        print("Training networks...")
        for learn_idx in range(self.num_learning_iterations):
            for _ in range(10):
                # print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))
                # Optimizers for critic and actor

                # Sample a random batch of trajectories from the replay buffer
                states, actions, log_probs, rewards = self.get_batch(self.episode_batch_size)

                # Train actor.
                self.actor.train()
                self.critic.eval()
                self.actor_opt.zero_grad()
                task_actions, task_log_probs = self.actor.predict(states, requires_grad=True)

                task_state_action_values = self.critic(self.get_critic_input(task_actions, states))
                actor_loss = self.actor_criterion(task_state_action_values, task_log_probs)
                actor_loss.backward()
                #nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
                self.actor_opt.step()

                # Train critic.
                self.critic.train()
                self.actor.eval()
                self.critic_opt.zero_grad()

                critic_input = self.get_critic_input(actions, states)
                state_trajectory_action_values = self.critic(critic_input)
                with torch.no_grad():
                    target_state_trajectory_action_values = self.target_critic(critic_input)
                    # TODO: Implement sampling for calculating expectation.
                    target_task_actions, _ = self.target_actor.predict(states)
                    target_expected_state_values = self.target_critic(self.get_critic_input(target_task_actions, states))
                    _, target_log_trajectory_task_action_probs = self.target_actor.predict(
                        states,
                        action=self.expand_actions(actions)
                    )
                critic_loss = self.critic_criterion(state_trajectory_action_values,
                                                    target_state_trajectory_action_values.detach(),
                                                    target_expected_state_values.detach(),
                                                    rewards,
                                                    log_probs,
                                                    target_log_trajectory_task_action_probs.detach())
                critic_loss.backward()
                #nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
                self.critic_opt.step()

                # Write to log.
                if self.writer:
                    self.writer.add_scalar('train/loss/actor', actor_loss, self.step_counter)
                    self.writer.add_scalar('train/loss/critic', critic_loss, self.step_counter)
                self.step_counter += 1
            self.update_targets()




