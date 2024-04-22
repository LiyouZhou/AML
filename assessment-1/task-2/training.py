#!/usr/bin/env python3

import sys
import gym
import pickle
import random
from typing import Callable
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from multiprocessing import Process, Lock

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
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

lock = Lock()


# from https://www.kaggle.com/code/deeplyai/super-mario-bros-with-stable-baseline3-ppo
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.clear()

    def clear(self):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # reward += max(0, info["x_pos"] - self.max_x)
        if (info["x_pos"] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            # reward -= 500
            done = True

        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]

        if self.current_x_count > 100:
            # reward -= 500
            done = True
            # print("STUCK")

        return state, reward/15, done, info


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


# policy_kwargs = dict(
#     features_extractor_class=MarioNet,
#     features_extractor_kwargs=dict(features_dim=512),
# )


class TrainAndLoggingCallback(BaseCallback):
    EPISODE_NUMBERS = 10
    MAX_TIMESTEP_TEST = 10000

    def __init__(
        self, check_freq, save_path, env, model, total_timestep_numb, lock, verbose=1
    ):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = Path(save_path)
        self.env = env
        self.model = model
        self.total_timestep_numb = total_timestep_numb
        self.lock = lock
        self.monitor_file_path = self.save_path / "monitor.csv"

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.lock.acquire()
            model_path = (
                self.save_path / "checkpoints" / "best_model_{}".format(self.n_calls)
            )
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

            self.lock.release()
        return True


# create the learning environment
def make_single_env(gym_id, seed, i=0, log_dir=None):
    env = gym_super_mario_bros.make(gym_id)
    RIGHT_JUMP_ONLY = [
        ["NOOP"],
        ["right"],
        ["right", "A"],
        ["right", "B"],
        ["right", "A", "B"],
    ]
    env = JoypadSpace(env, RIGHT_JUMP_ONLY)
    env = CustomRewardAndDoneEnv(env)
    env = atari_wrappers.MaxAndSkipEnv(env, 4)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.ClipRewardEnv(env)
    env = atari_wrappers.WarpFrame(env)
    env.seed(seed + i)
    env.action_space.seed(seed + i)
    env.observation_space.seed(seed + i)

    return env


def make_env(gym_id, seed, log_dir=None, n_envs=8):
    # env = DummyVecEnv([lambda: make_single_env(gym_id, seed, i, log_dir) for i in range(1)])
    env = make_vec_env(
        lambda: make_single_env(gym_id, seed, log_dir=log_dir),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=Path(log_dir) / "monitor" if log_dir else None,
        monitor_kwargs={"info_keywords": ("score", "x_pos", "y_pos", "life", "flag_get", "time")},
    )
    env = VecFrameStack(env, 4, channels_order="last")

    return env


def build_model(
    learningAlg,
    environment,
    seed,
    learning_rate,
    gamma,
    replay_buffer_size,
    exploration_fraction=0.9,
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
            exploration_fraction=exploration_fraction,
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
            tensorboard_log=log_dir,
        )
    elif learningAlg == "PPO":
        model = PPO(
            "CnnPolicy",
            environment,
            seed=seed,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
            # policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
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
    exploration_fraction=0.9,
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
        learningAlg,
        environment,
        seed,
        learning_rate,
        gamma,
        replay_buffer_size,
        exploration_fraction=exploration_fraction,
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
    exploration_fraction=0.9,
):
    # Format the date and time into a string
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = Path(f"logs/{timestamp}")
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
        "exploration_fraction": exploration_fraction,
    }
    seed = random.randint(0, 1000)
    params["seed"] = seed
    policyFileName = (
        learningAlg + "-" + environmentID + "-seed" + str(seed) + ".policy.pkl"
    )
    json.dump(params, open(log_folder / "params.json", "w"), indent=4)
    environment = make_env(environmentID, seed, log_dir=log_folder)
    model = build_model(
        learningAlg,
        environment,
        seed,
        learning_rate,
        gamma,
        replay_buffer_size,
        log_dir=log_folder,
        exploration_fraction=exploration_fraction,
    )

    CHECK_FREQ_NUMB = 10000
    callback = []
    # callback = TrainAndLoggingCallback(
    #     check_freq=CHECK_FREQ_NUMB,
    #     save_path=log_folder,
    #     env=environment,
    #     model=model,
    #     total_timestep_numb=num_training_steps,
    #     lock=lock,
    # )

    # train the agent or load a pre-trained one
    if trainMode:
        model.learn(
            total_timesteps=num_training_steps,
            tb_log_name="run",
            progress_bar=True,
            callback=callback,
        )
        print("Saving policy " + str(log_folder / policyFileName))
        pickle.dump(model.policy, open(log_folder / policyFileName, "wb"))
        model.save(str(log_folder / "final_model.zip"))
        eval(
            log_folder=log_folder,
            num_test_episodes=num_test_episodes,
            policy_rendering=policy_rendering,
        )


def eval(log_folder, num_test_episodes=10, policy_rendering=True):
    log_folder = Path(log_folder)
    policyFileName = list(log_folder.glob("*.zip"))[0]

    print("Loading policy...")
    param_file = policyFileName.parent / "params.json"
    with open(param_file, "r") as f:
        params = json.load(f)
        learningAlg = params["learningAlg"]
        seed = params["seed"]
        env = make_env(params["environmentID"], seed, log_dir=None, n_envs=1)
        learning_rate = params["learning_rate"]
        gamma = params["gamma"]
        replay_buffer_size = params["replay_buffer_size"]

    model = None
    if learningAlg == "DQN":
        model = DQN.load(policyFileName)
    elif learningAlg == "A2C":
        model = A2C.load(policyFileName)
    elif learningAlg == "PPO":
        model = PPO.load(policyFileName)
    print("Evaluating policy...")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_test_episodes * 5
    )

    eval_summary = f"EVALUATION: mean_reward={mean_reward}, std_reward={std_reward}"
    print(eval_summary)
    with open(log_folder / "eval.txt", "a") as f:
        print(eval_summary, file=f)

    steps_per_episode = 0
    reward_per_episode = 0
    total_cummulative_reward = 0
    step_count = 0
    episode = 1
    obs = env.reset()
    image_dir = log_folder / "policy_rendering"
    os.makedirs(image_dir, exist_ok=True)

    while True and policy_rendering:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps_per_episode += 1
        reward_per_episode += reward

        img = env.render("rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        step_count += 1
        cv2.imwrite(str(image_dir / f"{step_count}.png"), img)

        if any(done):
            episode_summary = f"episode={episode}, steps_per_episode={steps_per_episode}, reward_per_episode={reward_per_episode}"
            with open(log_folder / "eval.txt", "a") as f:
                print(episode_summary, file=f)
            print(episode_summary)
            total_cummulative_reward += reward_per_episode
            steps_per_episode = 0
            reward_per_episode = 0
            episode += 1
            obs = env.reset()

        if episode > num_test_episodes:
            final_summary = f"total_cummulative_reward={total_cummulative_reward}, avg_cummulative_reward={total_cummulative_reward / num_test_episodes}"
            with open(log_folder / "eval.txt", "a") as f:
                print(final_summary, file=f)
            print(final_summary)
            break

    env.close()

    cmd = f"ffmpeg -framerate 60 -i '{image_dir}/%d.png' -c:v libx264 -preset slow -tune stillimage -crf 24 -vf format=yuv420p -movflags +faststart -y {log_folder}/output.mp4"
    if policy_rendering:
        os.system(cmd)


if __name__ == "__main__":
    fire.Fire()
