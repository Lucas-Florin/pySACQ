import gym
import torch

# Add local files to path
from train import BaseTrainer
from networks_continuous import ContinuousActor, ContinuousCritic
from tasks import NoneScheduler
from learner import Learner
from sampler import Sampler


class PendulumTrainer(BaseTrainer):

    def init_task_scheduler(self):
        return NoneScheduler()

    def init_env(self):
        self.continuous = True
        return gym.make('Pendulum-v0')

    def get_nonlinear(self):
        non_linear = torch.nn.ELU()
        return non_linear

    def get_actor(self):
        return ContinuousActor(use_gpu=self.use_gpu)

    def get_critic(self):
        return ContinuousCritic(use_gpu=self.use_gpu)

    def get_sampler(self):
        return Sampler(self.actor, self.env, self.task, self.replay_buffer,
                       num_trajectories=self.args.num_trajectories,
                       task_period=30,
                       use_gpu=self.use_gpu,
                       continuous=True,
                       reward_scaling_factor=1 / 10,
                       writer=self.writer
                       )

    def get_learner(self):
        return Learner(self.actor, self.critic, self.task, self.replay_buffer,
                       num_learning_iterations=self.args.num_learning_iterations,
                       episode_batch_size=self.args.episode_batch_size,
                       use_gpu=self.use_gpu,
                       continuous=True,
                       writer=self.writer)

    @staticmethod
    def define_args():
        parser = BaseTrainer.define_args()

        return parser


if __name__ == '__main__':
    trainer = PendulumTrainer()

