import gym
import torch

# Add local files to path
from train import BaseTrainer
from networks_continuous import ContinuousActor, ContinuousCritic
from off_policy_learner import OffPolicyLearner
from actor_critic_networks import Actor, Critic
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
        return Actor(num_actions=1,
                  num_obs=3,
                  mean_scale=2,
                  std_low=0.1,
                  std_high=1,
                  action_bound=(-2,2))

    def get_critic(self):
        return Critic(num_actions=1, num_obs=3)

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
        TRAJECTORY_LENGTH = 200
        NUM_EVAL_TRAJECTORIES = 10
        NUM_TRAJECTORIES = 500
        BATCH_SIZE = 128
        NUM_TRAJECTORIES = 20
        BATCH_SIZE = 20
        UPDATE_TARGNETS_EVERY = 10
        NUM_TRAINING_ITERATIONS = 40
        TOTAL_TIMESTEPS = 1000
        ACTOR_LEARNING_RATE = 2e-4
        CRITIC_LEARNING_RATE = 2e-4
        GRADIENT_CLIPPING_VALUE = None
        NUM_EXPECTATION_SAMPLES = 1
        ENTROPY_REGULARIZATION_ON = False
        ENTROPY_REGULARIZATION = 1e-5
        ACTION_STD_LOW = 1e-1
        ACTION_STD_HIGH = 1
        ACTION_MEAN_SCALE = 2
        ACTION_BOUNDS = (-2, 2)
        REPLAY_BUFFER_SIZE = 10000
        LOG_EVERY = 10
        SAVE_MODEL_EVERY = 10
        return OffPolicyLearner(actor=self.actor,
                               critic=self.critic,
                               trajectory_length=TRAJECTORY_LENGTH,
                               actor_lr=ACTOR_LEARNING_RATE,
                               critic_lr=CRITIC_LEARNING_RATE,
                               expectation_samples=NUM_EXPECTATION_SAMPLES,
                               entropy_regularization_on=ENTROPY_REGULARIZATION_ON,
                               entropy_regularization=ENTROPY_REGULARIZATION,
                               gradient_clip_val=GRADIENT_CLIPPING_VALUE,
                               update_targnets_every=UPDATE_TARGNETS_EVERY,
                               num_training_iter=NUM_TRAINING_ITERATIONS,
                               minibatch_size=BATCH_SIZE,
                               logger=self.writer)

    @staticmethod
    def define_args():
        parser = BaseTrainer.define_args()

        return parser


if __name__ == '__main__':
    trainer = PendulumTrainer()

