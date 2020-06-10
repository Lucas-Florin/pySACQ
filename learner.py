import random
import torch

from losses.retrace_loss import RetraceLoss
from losses.actor_loss import ActorLoss


class Learner:

    def __init__(self,
                 actor,
                 critic,
                 task_scheduler,
                 replay_buffer,
                 num_learning_iterations=10,
                 episode_batch_size=10,
                 lr=0.0002,
                 writer=None):

        self.actor = actor
        self.critic = critic
        self.task_scheduler = task_scheduler
        self.replay_buffer = replay_buffer
        self.num_learning_iterations = num_learning_iterations
        self.episode_batch_size = episode_batch_size
        self.lr = lr
        self.writer = writer
        self.step_counter = 0

        # TODO: L2 regularization?
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr)

        # TODO: Specify alpha and gamma parameters.
        self.actor_criterion = ActorLoss()
        self.critic_criterion = RetraceLoss()

        # TODO: Implement copying models and set to eval.
        self.target_actor = actor
        self.target_critic = critic

    def update_targets(self):
        # TODO: Implement
        pass

    def learn(self):
        # TODO: Implement separate target actor and critic nets.
        self.actor.train()
        self.critic.train()
        for learn_idx in range(self.num_learning_iterations):
            # print('Learning: trajectory %s of %s' % (learn_idx + 1, num_learning_iterations))
            # Optimizers for critic and actor

            # TODO: Implement true batch learning.
            for batch_idx in range(self.episode_batch_size):
                # Sample a random trajectory from the replay buffer
                trajectory = random.choice(self.replay_buffer)
                states = trajectory.states
                rewards = trajectory.rewards
                actions = trajectory.actions
                log_probs = trajectory.log_probs
                num_steps = states.shape[0]

                # Train actor.
                self.actor.train()
                self.actor_opt.zero_grad()
                task_actions, task_log_probs = self.actor.predict(states)
                # TODO: Implement as separate function. Here or in critic class.
                # TODO: Allow for multidimensional actions.
                # TODO: Implement computing tasks separately.
                critic_input = torch.cat([task_actions.float(), states], dim=1)
                task_state_action_values = self.critic(critic_input)
                actor_loss = self.actor_criterion(task_state_action_values, task_log_probs)
                actor_loss.backward()
                self.actor_opt.step()

                # Train critic.
                self.critic.train()
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




