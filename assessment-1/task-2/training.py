#!/usr/bin/env python3

import sys
import gym
import pickle
import random
from typing import Callable
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from stable_baselines3.common import atari_wrappers
import fire
import datetime
import json
import os
from pathlib import Path
import cv2
import numpy as np

from gym.wrappers import GrayScaleObservation
import torch as th
from torch import nn

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# from https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo
class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=newshape, dtype=np.uint8
        )

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame


# from https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info["x_pos"] - self.max_x)
        if (info["x_pos"] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]

        if self.current_x_count > 100:
            reward -= 500
            done = True
            # print("STUCK")

        return state, reward / 10.0, done, info


class MarioNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MarioNet,
    features_extractor_kwargs=dict(features_dim=512),
)


class TrainAndLoggingCallback(BaseCallback):
    EPISODE_NUMBERS = 10
    MAX_TIMESTEP_TEST = 1000

    def __init__(
        self, check_freq, save_path, env, model, total_timestep_numb, verbose=1
    ):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = Path(save_path)
        self.env = env
        self.model = model
        self.total_timestep_numb = total_timestep_numb

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = self.save_path / "best_model_{}".format(self.n_calls)
            self.model.save(model_path)

            total_reward = [0] * self.EPISODE_NUMBERS
            total_time = [0] * self.EPISODE_NUMBERS
            best_reward = 0

            for i in range(self.EPISODE_NUMBERS):
                state = self.env.reset()  # reset for each new trial
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < self.MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = self.env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

                state = self.env.reset()  # reset for each new trial

            print("time steps:", self.n_calls, "/", self.total_timestep_numb)
            print(
                "average reward:",
                (sum(total_reward) / self.EPISODE_NUMBERS),
                "average time:",
                (sum(total_time) / self.EPISODE_NUMBERS),
                "best_reward:",
                best_reward,
            )

            with open(self.save_path / "reward_log.txt", "a") as f:
                print(
                    self.n_calls,
                    ",",
                    sum(total_reward) / self.EPISODE_NUMBERS,
                    ",",
                    best_reward,
                    file=f,
                )

        return True


# create the learning environment
def make_env(gym_id, seed, log_dir=None):
    env = gym_super_mario_bros.make(gym_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = atari_wrappers.MaxAndSkipEnv(env, 4)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.ClipRewardEnv(env)
    env = Monitor(env, filename=log_dir)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def build_model(
    learningAlg,
    environment,
    seed,
    learning_rate,
    gamma,
    replay_buffer_size,
    log_dir=None,
):
    # create the agent's model using one of the selected algorithms
    # note: exploration_fraction=0.9 means that it will explore 90% of the training steps
    print(f"Building model {learningAlg}...")
    if learningAlg == "DQN":
        model = DQN(
            "CnnPolicy",
            environment,
            seed=seed,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_size=replay_buffer_size,
            exploration_fraction=0.9,
            verbose=1,
            tensorboard_log=log_dir,
        )
    elif learningAlg == "A2C":
        model = A2C(
            "CnnPolicy",
            environment,
            seed=seed,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
        )
    elif learningAlg == "PPO":
        GAE = 1.0
        ENT_COEF = 0.01
        N_STEPS = 512
        GAMMA = 0.9
        BATCH_SIZE = 64
        N_EPOCHS = 10
        model = PPO(
            "CnnPolicy",
            environment,
            seed=seed,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gae_lambda=GAE,
            ent_coef=ENT_COEF,
        )
    else:
        print("UNKNOWN learningAlg=" + str(learningAlg))

    return model


def load_model(
    policyFileName,
    learningAlg=None,
    environment=None,
    seed=None,
    learning_rate=None,
    gamma=None,
    replay_buffer_size=None,
):
    # load params from json file if not specified
    if learningAlg is None:
        policyFileName = Path(policyFileName)
        # get folder of the path
        param_file = policyFileName.parent / "params.json"
        with open(param_file, "r") as f:
            params = json.load(f)
            learningAlg = params["learningAlg"]
            seed = params["seed"]
            environment = make_env(params["environmentID"], seed)
            learning_rate = params["learning_rate"]
            gamma = params["gamma"]
            replay_buffer_size = params["replay_buffer_size"]

    print("Loading policy...")
    with open(policyFileName, "rb") as f:
        policy = pickle.load(f)
    model = build_model(
        learningAlg, environment, seed, learning_rate, gamma, replay_buffer_size
    )
    model.policy = policy
    return model


def train(
    environmentID="SuperMarioBros2-v1",
    trainMode=True,  # if sys.argv[1] == "train" else False
    learningAlg="DQN",
    num_training_steps=500000,
    num_test_episodes=10,
    learning_rate=0.00083,
    gamma=0.995,
    policy_rendering=True,
    replay_buffer_size=30000,
):
    # Format the date and time into a string
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = f"logs/{timestamp}"
    os.makedirs(log_folder, exist_ok=True)

    params = {
        "environmentID": environmentID,
        "trainMode": trainMode,
        "learningAlg": learningAlg,
        "num_training_steps": num_training_steps,
        "num_test_episodes": num_test_episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "policy_rendering": policy_rendering,
        "replay_buffer_size": replay_buffer_size,
    }
    seed = random.randint(0, 1000)
    params["seed"] = seed
    policyFileName = (
        learningAlg + "-" + environmentID + "-seed" + str(seed) + ".policy.pkl"
    )
    environment = make_env(environmentID, seed, log_dir=log_folder)
    model = build_model(
        learningAlg,
        environment,
        seed,
        learning_rate,
        gamma,
        replay_buffer_size,
        log_dir=log_folder,
    )

    CHECK_FREQ_NUMB = 10000
    callback = TrainAndLoggingCallback(
        check_freq=CHECK_FREQ_NUMB,
        save_path=log_folder,
        env=environment,
        model=model,
        total_timestep_numb=num_training_steps,
    )

    # train the agent or load a pre-trained one
    if trainMode:
        model.learn(
            total_timesteps=num_training_steps,
            log_interval=num_training_steps / 30,
            tb_log_name=log_folder,
            progress_bar=True,
            callback=callback,
        )
        print("Saving policy " + str(log_folder + "/" + policyFileName))
        pickle.dump(model.policy, open(log_folder + "/" + policyFileName, "wb"))
        json.dump(params, open(log_folder + "/params.json", "w"), indent=4)
    else:
        print("Loading policy...")
        with open(policyFileName, "rb") as f:
            policy = pickle.load(f)
        model.policy = policy

    print("Evaluating policy...")
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=num_test_episodes * 5
    )
    print("EVALUATION: mean_reward=%s std_reward=%s" % (mean_reward, std_reward))


if __name__ == "__main__":
    fire.Fire(train)
