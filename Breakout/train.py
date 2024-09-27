import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import os
import sys
from torch.profiler import profile, record_function, ProfilerActivity, to_tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL import Atari_DQLAgent, BasicCNN, ExperienceReplay, Transition, Buffer, process_image
from tools import count_files, delete_oldest_file

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(i, u, env, criterion, optimizer, agent, replay: ExperienceReplay, dqn, 
               target_dqn, loss_buffer, q_buffer, gamma=0.9, batch_size=32, 
               update_period=8, target_update_period=500, double=True):
        
    observation, info = env.reset()
    dqn.train()
    s = to_tensor(observation, device=device)
    
    while True:
        action = agent.act(observation, device=device, crop=(50, None, None, None))
        observation, reward, terminated, truncated, info = env.step(action)
        s_prime = to_tensor(observation, device=device)
        transition = Transition(s, s_prime, action, reward, terminated)
        replay.push(transition)
        s = s_prime

        i += 1
        if i % update_period == 0:
            loss = update_dqn(dqn, target_dqn, criterion, optimizer, replay, q_buffer=q_buffer, 
                              batch_size=batch_size, gamma=gamma, double=double)
            loss_buffer.push(loss)
            if u % 1000 == 0:
                print(f'Update Step {u}')
            u += 1
        
        if u % target_update_period == 0:
            target_dqn.load_state_dict(dqn.state_dict()) # transfer weights from online network to target network
        
        done = terminated or truncated
        if done:
            break

    return i, u

def test_loop(env, agent, test_steps):
    # Unchanged
    observation, info = env.reset()
    eps = agent.epsilon
    agent.epsilon = 0.05 # ensure that the agent is acting greedily during test
    results = []

    episodes = 0
    steps = 0
    score = 0
    
    while steps < test_steps:
        observation, info = env.reset()
        while True:
            action = agent.act(observation, device=device, crop=(50, None, None, None))
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            score += reward
            done = terminated or truncated
            if done:
                break
        episodes += 1
    
    r_p_ep = score / episodes
    agent.epsilon = eps
    print(r_p_ep, score, episodes)
    
    return r_p_ep
    
def update_dqn(dqn, target_dqn, criterion, optimizer, replay: ExperienceReplay, 
                q_buffer, batch_size=32, gamma=0.9, double=True):
    s_batch, a_batch, r_batch, s_prime_batch, done_batch = replay.sample(batch_size, device=device)
    q_values = dqn(s_batch)
    
    if double: # online network used to select action, target network used to evaluate action
        q_values_prime = dqn(s_prime_batch)
        q_values_tgt = target_dqn(s_prime_batch)
        nxt_actions = torch.argmax(q_values_prime, dim=1)
        q_tgt = q_values_tgt[range(len(a_batch)), nxt_actions]
    else:
        q_values_tgt = target_dqn(s_prime_batch)
        q_tgt = torch.max(q_values_tgt, dim=1).values
    
    y = r_batch + gamma * q_tgt * (1 - done_batch)
    q = q_values[range(len(a_batch)), a_batch]
    q_buffer.push(torch.mean(q).item())
    optimizer.zero_grad()
    loss = criterion(q, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def main(args):
    env = gym.make('ALE/Breakout-v5', frameskip=1, full_action_space=False, obs_type=args.obs_type)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, terminal_on_life_loss=False)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    
    dqn = BasicCNN((80, 80), 4, 4).to(device)
    target_dqn = BasicCNN((80, 80), 4, 4).to(device)
    replay = ExperienceReplay(capacity=args.capacity)
    
    loss_buffer = Buffer()
    q_buffer = Buffer()
    test_reward_buffer = Buffer()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dqn.parameters(), lr=args.learning_rate)
    agent = Atari_DQLAgent(dqn, args.epsilon, 2)
    i, u = 0, 0 # i is the number of steps taken, u is the number of updates performed
    gamma = args.gamma
    epsilon = args.epsilon

    n_epochs = 0
    n_eps_updates = 0
    
    while u < args.max_updates:
        i, u, = train_loop(i, u, env, criterion, optimizer, loss_buffer=loss_buffer, q_buffer=q_buffer, 
                           agent=agent, replay=replay, dqn=dqn, target_dqn=target_dqn, 
                           batch_size=args.batch_size, update_period=args.update_period, 
                           target_update_period=args.target_update_period, gamma=gamma, double=args.double)
        
        if u // args.epoch_period > n_epochs:
            n_epochs += 1
            reward_per_ep = test_loop(env, agent, args.test_steps)
            test_reward_buffer.avg = reward_per_ep

            loss_buffer.update_long()
            q_buffer.update_long()
            test_reward_buffer.update_long()

            # plot statistics
            loss_buffer.plot_loss(log_scale=True, path='Breakout/plots/loss.png')
            q_buffer.plot_loss(path='Breakout/plots/q_vals.png')
            test_reward_buffer.plot_loss(path='Breakout/plots/test_reward.png')

            # save model
            torch.save(dqn.state_dict(), f'Breakout/models/DQN_Atari_ckpt_{n_epochs}_{epsilon}.pt')
            if count_files('Breakout/models') > args.max_models:
                delete_oldest_file('Breakout/models')

        if u // args.epsilon_decay_period > n_eps_updates:
            n_eps_updates += 1
            epsilon = max(epsilon-0.01, 0.1) # decrease in epsilon
            agent.epsilon = epsilon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN agent')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--update_period', type=int, default=4)
    parser.add_argument('--target_update_period', type=int, default=10000)
    parser.add_argument('--epoch_period', type=int, default=25000)
    parser.add_argument('--test_steps', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--epsilon_decay_period', type=int, default=5000, 
                        help='Number of updates between linear (-0.01) epsilon decrease')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma_tc', type=float, default=1)
    parser.add_argument('--capacity', type=int, default=int(5e5))
    parser.add_argument('--sequential', type=bool, default=False)
    parser.add_argument('--double', type=bool, default=True, help='Whether to use double DQN')
    parser.add_argument('--save_path', type=str, default='models/')
    parser.add_argument('--obs-type', type=str, default='grayscale')
    parser.add_argument('--max_updates', type=int, default=int(10e6))
    parser.add_argument('--max_models', type=int, default=5, help='Maximum number of models to store in models dir')

    args = parser.parse_args()
    main(args)