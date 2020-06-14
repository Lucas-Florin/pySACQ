# Import Python packages
import random
import copy

# Import libraries
import torch
import torch.nn as nn

# Import losses
from losses.retrace_loss import RetraceLoss
from losses.actor_loss import ActorLoss


class Learner:

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 task_scheduler,
                 replay_buffer: list,
                 num_learning_iterations=10,
                 episode_batch_size=10,
                 lr=0.0002,
                 use_gpu=False,
                 writer=None):

        self.actor = actor
        self.critic = critic
        self.task_scheduler = task_scheduler
        self.replay_buffer = replay_buffer
        self.num_learning_iterations = num_learning_iterations
        self.episode_batch_size = episode_batch_size
        self.lr = lr
        self.writer = writer

        self.use_gpu = use_gpu
        self.step_counter = 0

        self.actor_opt = torch.optim.Adam(actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr)

        # TODO: Specify alpha and gamma parameters.
        self.actor_criterion = ActorLoss()
        self.critic_criterion = RetraceLoss()

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor.eval()
        self.target_critic.eval()

    def update_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def learn(self):
        for learn_idx in range(self.num_learning_iterations):
            # print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))
            # Optimizers for critic and actor

            # TODO: Implement true batch learning.
            for batch_idx in range(self.episode_batch_size):
                # Sample a random trajectory from the replay buffer
                trajectory = random.choice(self.replay_buffer)
                states = trajectory.states.cuda() if self.use_gpu else trajectory.states
                rewards = trajectory.rewards.cuda() if self.use_gpu else trajectory.rewards
                actions = trajectory.actions.cuda() if self.use_gpu else trajectory.actions
                log_probs = trajectory.log_probs.cuda() if self.use_gpu else trajectory.log_probs
                num_steps = states.shape[0]

                # Train actor.
                self.actor.train()
                self.critic.eval()
                self.actor_opt.zero_grad()
                task_actions, task_log_probs = self.actor.predict(states)
                # TODO: Allow for multidimensional actions.
                critic_input = torch.cat([task_actions.float().unsqueeze(2),
                                          states.unsqueeze(1).expand(-1, self.actor.num_intentions, -1)], dim=2)
                task_state_action_values = self.critic(critic_input)
                actor_loss = self.actor_criterion(task_state_action_values, task_log_probs)
                actor_loss.backward()
                self.actor_opt.step()

                # Train critic.
                self.critic.train()
                self.actor.eval()
                self.critic_opt.zero_grad()

                critic_input = torch.cat([actions, states], dim=1)
                state_action_values = self.critic(critic_input)
                target_state_action_values = self.target_critic(critic_input).detach()
                critic_loss = self.critic_criterion(state_action_values,
                                                    target_state_action_values,
                                                    rewards,
                                                    log_probs,
                                                    task_log_probs.detach())
                critic_loss.backward()
                self.critic_opt.step()

                # Write to log.
                if self.writer:
                    self.writer.add_scalar('train/loss/actor', actor_loss, self.step_counter)
                    self.writer.add_scalar('train/loss/critic', critic_loss, self.step_counter)
                self.step_counter += 1
        self.update_targets()




