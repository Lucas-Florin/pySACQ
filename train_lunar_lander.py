import gym
import torch

# Add local files to path
from train import BaseTrainer
from networks import Actor, Critic
from tasks import TaskScheduler
from learner import Learner
from sampler import Sampler


class LunarLanderTrainer(BaseTrainer):

    def init_task_scheduler(self):
        return TaskScheduler()

    def init_env(self):
        return gym.make('LunarLander-v2')

    def get_nonlinear(self):
        if self.args.non_linear == 'relu':
            non_linear = torch.nn.ReLU()
        elif self.args.non_linear == 'elu':
            non_linear = torch.nn.ELU()
        else:
            raise ValueError('Invalid name for nonlinear function. ')
        return non_linear

    def get_actor(self):
        return Actor(use_gpu=self.use_gpu, non_linear=self.non_linear, batch_norm=self.args.batch_norm)

    def get_critic(self):
        return Critic(use_gpu=self.use_gpu, non_linear=self.non_linear, batch_norm=self.args.batch_norm)

    def get_sampler(self):
        return Sampler(self.actor, self.env, self.task, self.replay_buffer,
                       num_trajectories=self.args.num_trajectories,
                       task_period=30,
                       use_gpu=self.use_gpu,
                       writer=self.writer
                       )

    def get_learner(self):
        return Learner(self.actor, self.critic, self.task, self.replay_buffer,
                       num_learning_iterations=self.args.num_learning_iterations,
                       episode_batch_size=self.args.episode_batch_size,
                       use_gpu=self.use_gpu,
                       writer=self.writer)

    @staticmethod
    def define_args():
        parser = BaseTrainer.define_args()
        # Model parameters
        parser.add_argument('--non_linear', type=str, default='relu', help='Non-linearity in the nets [default: ReLU]')
        parser.add_argument('--batch_norm', dest='batch_norm', default=False, action='store_true',
                            help='Batch norm applied to input layers [default: False]')

        return parser


if __name__ == '__main__':
    trainer = LunarLanderTrainer()

