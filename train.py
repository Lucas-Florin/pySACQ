import argparse
from pathlib import Path
import sys
import time
import os
from collections import deque

import gym
import torch
import numpy as np
from tensorboardX import SummaryWriter

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))

# Log and model saving parameters

class BaseTrainer:

    def __init__(self):
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S_")
        self.args = self.get_args()
        self.use_gpu = self.check_gpu()

        # Replay buffer stores collected trajectories
        self.replay_buffer = deque(maxlen=self.args.buffer_size)

        # Environment is the lunar lander from OpenAI gym

        self.continuous = False
        self.reward_scaling_factor = 1.0
        self.env = self.init_env()
        self.transform = self.get_transform()

        # task scheduler is defined in tasks.py
        self.task = self.init_task_scheduler()

        # Write tensorboard logs to local logs folder
        self.writer = None
        if self.args.log:
            # TODO: Optimize Tensorboard.
            log_dir = root_dir / 'logs' / self.args.log
            self.writer = SummaryWriter(log_dir=str(log_dir))

        self.test_step = 0

        # Non-linearity is an argument
        self.non_linear = self.get_nonlinear()

        # Actor and Critic networks
        actor = self.get_actor()
        critic = self.get_critic()
        self.actor = actor.cuda() if self.use_gpu else actor
        self.critic = critic.cuda() if self.use_gpu else critic

        if self.args.model:  # TEST MODE
            self.use_gpu = False
            self.load_models(self.actor, self.critic)
            self.evaluate()

        else:  # TRAIN MODE
            print('Train mode. ')

            self.learner = self.get_learner()
            self.sampler = self.get_sampler()

            self.train()

            # Save the model to local directory
            if self.args.saveas is not None:
                self.save()

        # Close writer
        if self.writer is not None:
            self.writer.close()

    def save(self):
        save_path = str(root_dir / 'models' / self.args.saveas)
        Path(str(root_dir / 'models/')).mkdir(parents=True, exist_ok=True)
        print('Saving models to %s' % save_path)
        torch.save(self.actor.state_dict(), save_path + '_actor.pt')
        torch.save(self.critic.state_dict(), save_path + '_critic.pt')
        print('...done')

    def init_task_scheduler(self):
        raise NotImplementedError

    def init_env(self):
        raise NotImplementedError

    def get_nonlinear(self):
        raise NotImplementedError

    def get_actor(self):
        raise NotImplementedError

    def get_critic(self):
        raise NotImplementedError

    def get_sampler(self):
        raise NotImplementedError

    def get_learner(self):
        raise NotImplementedError

    def get_transform(self):
        return None

    def load_models(self, actor, critic):
        model_path = str(root_dir / 'models' / self.args.model)
        print('Loading models from %s' % model_path)
        actor_sd = torch.load(model_path + '_actor.pt', map_location=torch.device('cpu'))
        actor.load_state_dict(actor_sd)
        critic_sd = torch.load(model_path + '_critic.pt', map_location=torch.device('cpu'))
        critic.load_state_dict(critic_sd)
        print('...done')
        return actor, critic

    def evaluate(self):
        while True:
            self.run(render=self.args.render)

    def train(self):
        print('Start training. ')
        for i in range(self.args.num_train_cycles):
            t = time.time()
            print('Training cycle %s of %s' % (i, self.args.num_train_cycles))
            self.sampler.sample()
            print("Sampling: {:.2f}s".format(time.time() - t))
            t = time.time()
            self.learner.learn()
            print("Learning: {:.2f}s".format(time.time() - t))
            t = time.time()
            for _ in range(5):
                self.run()
            if (i+1) % self.args.save_freq == 0:
                self.save()
            print("Evaluation: {:.2f}s".format(time.time() - t))
            t = time.time()

    def check_gpu(self):
        # Make sure we can use gpu
        use_gpu = self.args.use_gpu

        if use_gpu:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu_device
        use_gpu = torch.cuda.is_available() and self.args.use_gpu
        print('Use GPU: %s' % use_gpu)

        return use_gpu

    def get_args(self):
        # Parse and print out parameters

        print('Timestamp: {}'.format(self.timestamp))
        args = self.define_args().parse_args()
        args.log = self.timestamp + args.log if args.log is not None else None
        args.saveas = self.timestamp + args.saveas if args.saveas is not None else None
        print('Running Trainer. Parameters:')
        for attr, value in args.__dict__.items():
            print('%s : %s' % (attr.upper(), value))

        return args

    @staticmethod
    def define_args():
        parser = argparse.ArgumentParser(description='Train Arguments')
        parser.add_argument('--log', type=str, default=None,
                            help='Write tensorboard style logs to this folder [default: None]')
        parser.add_argument('--saveas', type=str, default=None, help='savename for model (Training) [default: None]')
        parser.add_argument('--model', type=str, default=None, help='savename for model (Evaluating) [default: None]')
        parser.add_argument('--render', action='store_true',
                            help='render the environment for display')

        # Training parameters
        parser.add_argument('--use-gpu', action='store_true',
                            help='use GPU for training')
        parser.add_argument('--gpu-device', default='0', type=str,
                            help='gpu device ids for CUDA_VISIBLE_DEVICES')
        parser.add_argument('--num_train_cycles', type=int, default=1000, help='Number of training cycles [default: 1]')
        parser.add_argument('--num_trajectories', type=int, default=5,
                            help='Number of trajectories collected per acting cycle [default: 5]')
        parser.add_argument('--num_learning_iterations', type=int, default=1,
                            help='Number of learning iterations per learn cycle [default: 1]')
        parser.add_argument('--episode_batch_size', type=int, default=16,
                            help='Number of trajectories per batch (gradient push) [default: 2]')
        parser.add_argument('--buffer_size', type=int, default=200,
                            help='Number of trajectories in replay buffer [default: 200]')
        parser.add_argument('--save-freq', type=int, default=10,
                            help='Save frequency')

        return parser

    def run(self, min_rate=0.02, render=False):
        """
        Runs the actor policy on the environment, rendering it. This does not store anything
        and is only used for visualization.
        :param min_rate: (float) minimum framerate
        :param render: (Bool) toggle for rendering to window
        :return: None
        """
        obs = self.env.reset()
        done = False
        # Counter variables for number of steps and total episode time
        epoch_tic = time.clock()
        num_steps = 0
        reward = 0
        rewards = list()
        with torch.no_grad():
            while not done:
                if self.transform is not None:
                    obs = self.transform(obs)
                step_tic = time.clock()
                if render:
                    self.env.render()
                # Use the previous observation to get an action from policy
                self.actor.eval()
                obs = torch.tensor(obs, dtype=torch.float)
                obs = obs.cuda() if self.use_gpu else obs
                action, _ = self.actor.predict(obs, task=-1, noise=False)  # Last intention is main task
                # Step the environment and push outputs to policy
                gym_action = action.detach().cpu().squeeze()
                if self.continuous and gym_action.dim() == 0:
                    gym_action = gym_action.unsqueeze(0)
                obs, reward, done, _ = self.env.step(gym_action.numpy())
                rewards.append(reward)

                step_toc = time.clock()
                step_time = step_toc - step_tic
                if render and min_rate and step_time < min_rate:  # Sleep to ensure minimum rate
                    time.sleep(min_rate - step_time)
                num_steps += 1
        self.test_step += 1
        # Total elapsed time in epoch
        epoch_toc = time.clock()
        epoch_time = epoch_toc - epoch_tic
        print('Evaluation mean reward %s ' % (np.mean(rewards)))
        if self.writer:
            self.writer.add_scalar('test/reward', np.mean(rewards), self.test_step)


if __name__ == '__main__':
    trainer = BaseTrainer()

