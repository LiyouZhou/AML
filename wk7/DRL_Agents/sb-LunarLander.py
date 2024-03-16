#####################################################
# sb-LunarLander.py
#
# This program trains deep reinforcement learning agents to play the game of
# SuperMarioBros2 via trial-and-error. See details in the following link:
# https://github.com/Kautenja/gym-super-mario-bros
#
# Although the dependencies have been installed in the PCs of the labs. You may
# require the following in case of installation in your own PCs:
# pip install nes-py
# pip install gym-super-mario-bros==7.3.0
# pip install setuptools==65.5.0 "wheel<0.40.0"
# pip install gym==0.21.0
# pip install gymnasium
# pip install stable-baselines3[extra]==1.8.0
# pip install sb3_contrib==1.8.0

# Version: 1.0 -- version tested with DQN, A2C and PPO
# Date: 10 March 2024
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import sys, cv2
import gym
import pickle
import random
from typing import Callable
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("USAGE: sb-LunarLander-v2.py (train|test) (DQN|A2C|PPO) [seed number]")
    exit(0)

environmentID = "LunarLander-v2"
trainMode = True if sys.argv[1] == "train" else False
learningAlg = sys.argv[2]
seed = random.randint(0, 1000) if trainMode else int(sys.argv[3])
policyFileName = learningAlg + "-" + environmentID + "-seed" + str(seed) + ".policy.pkl"
num_training_steps = 500000
num_test_episodes = 10
learning_rate = 0.00083
gamma = 0.995
policy_rendering = True

# define learning environment and model
environment = gym.make(environmentID)
if learningAlg == "DQN":
    model = DQN(
        "MlpPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=50000,
        exploration_fraction=0.9,
        verbose=1,
    )
elif learningAlg == "A2C":
    model = A2C(
        "MlpPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
    )
elif learningAlg == "PPO":
    model = PPO(
        "MlpPolicy",
        environment,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
    )
else:
    print("UNKNOWN learningAlg=" + str(learningAlg))
    exit(0)

print(model.policy)

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

# visualise the agent's learnt behaviour (subject to dependencies being compatible)
steps_per_episode = 0
reward_per_episode = 0
total_cummulative_reward = 0
episode = 1
env = model.get_env()
obs = env.reset()
step = 0
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
    # img = env.render( mode='rgb_array')
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # step += 1
    # cv2.imwrite(f"{learningAlg}_test/{step}.png", img)
    if episode > num_test_episodes:
        print(
            "total_cummulative_reward=%s avg_cummulative_reward=%s"
            % (total_cummulative_reward, total_cummulative_reward / num_test_episodes)
        )
        break
env.close()


# DQN
# EVALUATION: mean_reward=189.30313694 std_reward=83.91632902643407
# episode=1, steps_per_episode=245, reward_per_episode=[230.39824]
# episode=2, steps_per_episode=419, reward_per_episode=[220.05057]
# episode=3, steps_per_episode=183, reward_per_episode=[235.09819]
# episode=4, steps_per_episode=929, reward_per_episode=[94.44079]
# episode=5, steps_per_episode=190, reward_per_episode=[246.07018]
# episode=6, steps_per_episode=341, reward_per_episode=[-186.68533]
# episode=7, steps_per_episode=240, reward_per_episode=[251.87146]
# episode=8, steps_per_episode=1000, reward_per_episode=[138.8898]
# episode=9, steps_per_episode=1000, reward_per_episode=[147.935]
# episode=10, steps_per_episode=1000, reward_per_episode=[127.15026]
# total_cummulative_reward=[1505.2191] avg_cummulative_reward=[150.52191]

# A2C
# EVALUATION: mean_reward=44.970668399999994 std_reward=136.1680263230013
# episode=1, steps_per_episode=221, reward_per_episode=[-76.64818]
# episode=2, steps_per_episode=215, reward_per_episode=[-82.371994]
# episode=3, steps_per_episode=159, reward_per_episode=[-78.24884]
# episode=4, steps_per_episode=355, reward_per_episode=[206.27112]
# episode=5, steps_per_episode=357, reward_per_episode=[-67.16267]
# episode=6, steps_per_episode=364, reward_per_episode=[247.20294]
# episode=7, steps_per_episode=319, reward_per_episode=[-127.031685]
# episode=8, steps_per_episode=396, reward_per_episode=[-149.29202]
# episode=9, steps_per_episode=273, reward_per_episode=[-3.4292984]
# episode=10, steps_per_episode=177, reward_per_episode=[-71.33786]
# total_cummulative_reward=[-202.04848] avg_cummulative_reward=[-20.204847]

# PPO
# EVALUATION: mean_reward=276.97758483999996 std_reward=29.176939479045444
# episode=1, steps_per_episode=196, reward_per_episode=[309.35522]
# episode=2, steps_per_episode=203, reward_per_episode=[280.1826]
# episode=3, steps_per_episode=216, reward_per_episode=[289.4242]
# episode=4, steps_per_episode=232, reward_per_episode=[252.0124]
# episode=5, steps_per_episode=235, reward_per_episode=[261.45447]
# episode=6, steps_per_episode=226, reward_per_episode=[254.21779]
# episode=7, steps_per_episode=254, reward_per_episode=[274.5995]
# episode=8, steps_per_episode=225, reward_per_episode=[297.87915]
# episode=9, steps_per_episode=206, reward_per_episode=[272.9366]
# episode=10, steps_per_episode=234, reward_per_episode=[250.21104]
# total_cummulative_reward=[2742.273] avg_cummulative_reward=[274.2273]
