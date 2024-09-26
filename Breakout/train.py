import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL import Atari_DQLAgent, BasicCNN, ExperienceReplay, Transition, LossBuffer, process_image

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(i, u, env, criterion, optimizer, agent, replay: ExperienceReplay, dqn, target_dqn, loss_buffer, gamma=0.9,
               batch_size=32, update_period=8, target_update_period=500, double=True):
        
        observation, info = env.reset()
        dqn.train()
        s = process_image(observation, device=device, crop_top=50)
        
        while True:
            action = agent.act(observation, device=device, crop=(50, None, None, None))
            observation, reward, terminated, truncated, info = env.step(action)

            s_prime = process_image(observation, device=device, crop_top=50)
            transition = Transition(s, s_prime, action, reward, terminated)
            replay.push(transition)
            s = s_prime


            i += 1
            if i % update_period == 0:
                loss = update_dqn(dqn, target_dqn, criterion, optimizer, replay, batch_size=batch_size, gamma=gamma, double=double)
                loss_buffer.push(loss)
                if u % 10 == 0:
                    print(f'Update Step {u}')
                u += 1
            if u % target_update_period == 0:
                target_dqn.load_state_dict(dqn.state_dict()) # transfer weights from online network to target network
            
            done = terminated or truncated
            if done:
                break

        return i, u


def test_loop(env, agent, test_episodes=100):
    observation, info = env.reset()
    eps = agent.epsilon
    agent.epsilon = 0.01 # ensure that the agent is acting greedily during test
    results = []
    
    for _ in range(test_episodes):
        observation, info = env.reset()
        score = 0
        while True:
            action = agent.act(observation, device=device, crop=(50, None, None, None))
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            print(info['lives'], score)
            
            done = terminated or truncated
            if done:
                break
        results.append(score)
    
    agent.epsilon = eps
    return results
    
                

def update_dqn(dqn, target_dqn, criterion, optimizer, replay: ExperienceReplay, batch_size=32, gamma=0.9, double=True):
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
    optimizer.zero_grad()
    loss = criterion(q, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    env = gym.make('Breakout-v0')

    dqn = BasicCNN((80, 80), 3, 4).to(device)
    target_dqn = BasicCNN((80, 80), 3, 4).to(device)
    replay = ExperienceReplay(capacity=args.capacity)
    loss_buffer = LossBuffer()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dqn.parameters(), lr=args.learning_rate)
    agent = Atari_DQLAgent(dqn, args.epsilon, 2)
    i, u = 0, 0 # i is the number of steps taken, u is the number of updates performed
    gamma = args.gamma
    epsilon = args.epsilon
    
    for ep in range(int(args.episodes)):
        i, u = train_loop(i, u, env, criterion, optimizer, loss_buffer=loss_buffer, agent=agent, replay=replay, dqn=dqn, target_dqn=target_dqn, 
                          batch_size=args.batch_size, update_period=args.update_period, 
                          target_update_period=args.target_update_period, gamma=gamma, double=args.double)
        
        if ep % args.test_period == 0:
            print(f'Testing')
            score = test_loop(env, agent, args.test_episodes)
            print(f'Test after Episode {ep+1} > Score: {score}')
        
        if ep % args.lr_decay_period == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.2

        epsilon = max(epsilon * args.epsilon_decay, 0.1)
        #gamma = max((0.9 - args.gamma_tc**(ep+1)), 0)
        agent.epsilon = epsilon
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN agent')
    parser.add_argument('--batch_size', type=int, default=128, help='path to the pretrained model')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--update_period', type=int, default=8)
    parser.add_argument('--target_update_period', type=int, default=1000)
    parser.add_argument('--test_period', type=int, default=2)
    parser.add_argument('--test_episodes', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_period', type=int, default=30000)
    parser.add_argument('--gamma_tc', type=float, default=1)
    parser.add_argument('--capacity', type=float, default=10000)
    parser.add_argument('--episodes', type=float, default=100)
    parser.add_argument('--sequential', type=bool, default=False)
    parser.add_argument('--double', type=bool, default=True, help='Whether to use double DQN')
    parser.add_argument('--save_path', type=str, default='models/')

    args = parser.parse_args()
    main(args)