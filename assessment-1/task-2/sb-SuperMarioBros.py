#####################################################
# sb-LunarLander.py
#
# This program trains deep reinforcement learning agents to solve the decision-making
# problem of the the so-called LunarLander-v2. See details in the following link:
# https://www.gymlibrary.dev/environments/box2d/lunar_lander/
#
# Although the dependencies have been installed in the PCs of the labs. You may
# should be aware the following in case of installation in your own PCs:
# pip install nes-py
# pip install gym-super-mario-bros==7.3.0
# pip install setuptools==65.5.0 "wheel<0.40.0"
# pip install gym==0.21.0
# pip install gymnasium
# pip install stable-baselines3[extra]==1.8.0
# pip install sb3_contrib==1.8.0

# Version: 1.0 -- extended version from sb-LunarLander.py
# Date: 10 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import sys
import gym
import pickle
import random
from typing import Callable
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from stable_baselines3.common import atari_wrappers


if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("USAGE: sb-SuperMarioBros2-v1.py (train|test) (DQN|A2C|PPO) [seed_number]")
    exit(0)

environmentID = "SuperMarioBros2-v1"
trainMode = True if sys.argv[1] == "train" else False
learningAlg = sys.argv[2]
seed = random.randint(0, 1000) if trainMode else int(sys.argv[3])
policyFileName = learningAlg + "-" + environmentID + "-seed" + str(seed) + ".policy.pkl"
num_training_steps = 500000
num_test_episodes = 10
learning_rate = 0.00083
gamma = 0.995
policy_rendering = True


# create the learning environment
def make_env(gym_id, seed):
    env = gym_super_mario_bros.make(gym_id)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = atari_wrappers.MaxAndSkipEnv(env, 4)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.ClipRewardEnv(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


environment = make_env(environmentID, seed)

# create the agent's model using one of the selected algorithms
# note: exploration_fraction=0.9 means that it will explore 90% of the training steps
if learningAlg == "DQN":
    model = DQN(
        "CnnPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=30000,
        exploration_fraction=0.9,
        verbose=1,
    )
elif learningAlg == "DQN":
    model = A2C(
        "CnnPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
    )
elif learningAlg == "DQN":
    model = PPO(
        "CnnPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
    )
else:
    print("UNKNOWN learningAlg=" + str(learningAlg))
    exit(0)

# train the agent or load a pre-trained one
if trainMode:
    model.learn(total_timesteps=num_training_steps, progress_bar=True)
    print("Saving policy " + str(policyFileName))
    pickle.dump(model.policy, open(policyFileName, "wb"))
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

# visualise the agent's learnt behaviour
steps_per_episode = 0
reward_per_episode = 0
total_cummulative_reward = 0
episode = 1
env = model.get_env()
obs = env.reset()
while True and policy_rendering:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps_per_episode += 1
    reward_per_episode += reward
    if any(done):
        print(
            "episode=%s, steps_per_episode=%s, reward_per_episode=%s"
            % (episode, steps_per_episode, reward_per_episode)
        )
        total_cummulative_reward += reward_per_episode
        steps_per_episode = 0
        reward_per_episode = 0
        episode += 1
        obs = env.reset()
    env.render("human")
    if episode > num_test_episodes:
        print(
            "total_cummulative_reward=%s avg_cummulative_reward=%s"
            % (total_cummulative_reward, total_cummulative_reward / num_test_episodes)
        )
        break
env.close()
