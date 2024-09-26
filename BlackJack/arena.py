import gymnasium as gym
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL import RandomAgent, DQLAgent



def main(args):
    episode_count = 0
    results = []

    if args.render:
        env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human')
    else:
        env = gym.make('Blackjack-v1', natural=False, sab=False)

    if args.agent == "random":
        agent = RandomAgent(a_size=2)
    elif args.agent == "dql":
        agent = DQLAgent(a_size=2)
    
    while episode_count < args.max_episodes:
        observation = env.reset()
        
        done = False
        
        while not done:
            action = agent.act(observation)
            observation, reward, done, _, _ = env.step(action)

        results.append(reward)
        episode_count += 1

    wins = results.count(1)
    losses = results.count(-1)
    draws = results.count(0)
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="random", help="Agent to use for playing the game")
    parser.add_argument("--max_episodes", type=int, default=100, help="Number of episodes to play")
    parser.add_argument("--render", type=bool, default=False, help="Render the game")
    args = parser.parse_args()

    main(args)