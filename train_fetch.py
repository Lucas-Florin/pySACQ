import gym
import torch

# Add local files to path
from train import BaseTrainer
from networks_continuous import ContinuousActor, ContinuousCritic
from tasks import NoneScheduler
from learner import Learner
from sampler import Sampler
from transforms import FetchTransform


class FetchTrainer(BaseTrainer):

    def init_task_scheduler(self):
        return NoneScheduler()

    def init_env(self):
        self.continuous = True
        return gym.make('FetchReach-v1', reward_type='dense')

    def get_nonlinear(self):
        non_linear = torch.nn.ELU()
        return non_linear

    def get_actor(self):
        return ContinuousActor(
            state_dim=13,
            base_hidden_size=32,
            head_input_size=16,
            head_hidden_size=8,
            action_dim=4,
            use_gpu=self.use_gpu
        )

    def get_critic(self):
        return ContinuousCritic(
            state_dim=17,
            base_hidden_size=64,
            head_input_size=64,
            head_hidden_size=32,
            head_output_size=1,
            use_gpu=self.use_gpu
        )

    def get_sampler(self):
        return Sampler(self.actor, self.env, self.task, self.replay_buffer,
                       num_trajectories=self.args.num_trajectories,
                       task_period=100,
                       use_gpu=self.use_gpu,
                       continuous=True,
                       reward_scaling_factor=1.0,
                       skip_steps=1,
                       transform=self.transform,
                       writer=self.writer
                       )

    def get_transform(self):
        return FetchTransform()

    def get_learner(self):
        return Learner(self.actor, self.critic, self.replay_buffer,
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
    trainer = FetchTrainer()
