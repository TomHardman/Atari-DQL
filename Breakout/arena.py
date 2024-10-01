import gymnasium as gym
import sys
import os
import torch
import numpy as np
import gc

import psutil
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL import RandomAgent, Atari_DQLAgent, process_image, BasicCNN
import threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

def render_env(env):
    while True:
        env.render()

def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    return mem

def main(args):
    episode_count = 0
    results = []

    if args.render:
        env = gym.make('Breakout-v4', full_action_space=False, frameskip=1, 
                       obs_type='grayscale', render_mode='human')
    else:
        env = gym.make('Breakout-v4', full_action_space=False, frameskip=1, 
                       obs_type='grayscale')
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, terminal_on_life_loss=True, frame_skip=4)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    if args.agent == "random":
            agent = RandomAgent(a_size=4)
    elif args.agent == "dql":
        dqn = BasicCNN((84, 84), 4, 4, conv3=True).to(device)
        agent = Atari_DQLAgent(a_size=4, epsilon=0.0, q_network=dqn)
        dqn.load_state_dict(torch.load("Breakout/models/DQN_Atari_ckpt_99_0.100_141.000.pt", map_location=device))
    
    while episode_count < args.max_episodes:
        observation, info = env.reset()

        first_state = True
        
        done = False
        score = 0
        
        while not done:
            if first_state:
                action = 1
                first_state = False
            else:
                action = agent.act(observation, device=device)
            observation, reward, life_loss, _, info = env.step(action)

            if life_loss:
                first_state = True
                if info['lives'] == 0:
                    done = True


            score += reward
        results.append(score)
        episode_count += 1
        print(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default='dql', help="Agent to use for playing the game")
    parser.add_argument("--max_episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--render", type=bool, default=True, help="Render the game")
    args = parser.parse_args()

    main(args)