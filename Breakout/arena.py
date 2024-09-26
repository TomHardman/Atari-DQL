import gymnasium as gym
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL import RandomAgent, Atari_DQLAgent, process_image



def main(args):
    episode_count = 0
    results = []

    if args.render:
        env = gym.make('Breakout-v0', render_mode='human')
    else:
        env = gym.make('Breakout-v0')

    if args.agent == "random":
        agent = RandomAgent(a_size=4)
    elif args.agent == "dql":
        agent = Atari_DQLAgent(a_size=4)
    
    while episode_count < args.max_episodes:
        observation, info = env.reset()
        
        done = False
        
        while not done:
            action = agent.act(observation)
            observation, reward, done, _, info = env.step(action)
            observation = torch.Tensor(observation)
            observation = process_image(observation)

        results.append(reward)
        episode_count += 1
    print(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="random", help="Agent to use for playing the game")
    parser.add_argument("--max_episodes", type=int, default=100, help="Number of episodes to play")
    parser.add_argument("--render", type=bool, default=True, help="Render the game")
    args = parser.parse_args()

    main(args)