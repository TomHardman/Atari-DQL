import random
import numpy as np
import torch
from .utils import process_image


class RandomAgent:
    def __init__(self, a_size: int) -> None:
        self.a_size = a_size

    def act(self, observation):
        return random.randint(0, self.a_size - 1)


class BJ_DQLAgent():
    def __init__(self, q_network, epsilon, a_size) -> None:
        self.q_network = q_network
        self.epsilon = epsilon
        self.a_size = a_size

    def act(self, observation):
        self.q_network.eval()
        observation = torch.Tensor(observation)
        
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.a_size - 1)
        
        else:
            with torch.no_grad():
                q_values = self.q_network.forward(observation)
                self.q_network.train()
                action = torch.argmax(q_values).item()
                return action


class Atari_DQLAgent():
    def __init__(self, q_network, epsilon, a_size) -> None:
        self.q_network = q_network
        self.epsilon = epsilon
        self.a_size = a_size

    def act(self, observation, crop):
        self.q_network.eval()
        observation = process_image(observation, crop_top=crop[0], crop_bottom=crop[1], 
                                    crop_left=crop[2], crop_right=crop[3])
        
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.a_size - 1)
        
        else:
            with torch.no_grad():
                q_values = self.q_network.forward(observation)
                self.q_network.train()
                action = torch.argmax(q_values).item()
                return action